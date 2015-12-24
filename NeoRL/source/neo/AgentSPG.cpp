#include "AgentSPG.h"

using namespace neo;

void AgentSPG::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange, float initThreshold,
	std::mt19937 &rng)
{
	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	cl_int2 prevLayerSize = inputSize;

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs(2);

		scDescs[0]._size = prevLayerSize;
		scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
		scDescs[0]._ignoreMiddle = false;
		scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
		scDescs[0]._weightLambda = _layerDescs[l]._scWeightLambda;
		scDescs[0]._useTraces = true;

		scDescs[1]._size = _layerDescs[l]._hiddenSize;
		scDescs[1]._radius = _layerDescs[l]._recurrentRadius;
		scDescs[1]._ignoreMiddle = true;
		scDescs[1]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
		scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
		scDescs[1]._useTraces = true;

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._hiddenSize, _layerDescs[l]._lateralRadius, initWeightRange, initThreshold, rng);

		_layers[l]._modulatedFeedForwardInput = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerSize.x, prevLayerSize.y);

		_layers[l]._modulatedRecurrentInput = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y);

		std::vector<PredictorSwarm::VisibleLayerDesc> predDescs;

		if (l < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[l]._hiddenSize;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;

			predDescs[1]._size = _layerDescs[l]._hiddenSize; // Same size as current layer
			predDescs[1]._radius = _layerDescs[l]._feedBackRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[l]._hiddenSize;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;
		}

		_layers[l]._predAction.createRandom(cs, program, predDescs, prevLayerSize, initWeightRange, rng);
		_layers[l]._predAttentionFeedForward.createRandom(cs, program, predDescs, prevLayerSize, initWeightRange, rng);
		_layers[l]._predAttentionRecurrent.createRandom(cs, program, predDescs, _layerDescs[l]._hiddenSize, initWeightRange, rng);

		// Create baselines
		_layers[l]._baseLines = createDoubleBuffer2D(cs, _layerDescs[l]._hiddenSize, CL_R, CL_FLOAT);

		_layers[l]._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y);

		_layers[l]._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

		if (l != 0) {
			cl::array<cl::size_type, 3> actionRegion = { _layers[l]._predAction.getHiddenSize().x, _layers[l]._predAction.getHiddenSize().y, 1 };

			_layers[l]._inhibitedAction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layers[l]._predAction.getHiddenSize().x, _layers[l]._predAction.getHiddenSize().y);

			cs.getQueue().enqueueFillImage(_layers[l]._inhibitedAction, zeroColor, zeroOrigin, actionRegion);
		}

		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._baseLines[_back], zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._reward, zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
	}

	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { actionSize.x, actionSize.y, 1 };

		_action = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), actionSize.x, actionSize.y);

		cs.getQueue().enqueueFillImage(_action, zeroColor, zeroOrigin, layerRegion);
	}

	_baseLineUpdateKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdate");
	_baseLineUpdateSumErrorKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdateSumError");
	_inhibitKernel = cl::Kernel(program.getProgram(), "phInhibit");
	_modulateKernel = cl::Kernel(program.getProgram(), "phModulate");
	_copyActionKernel = cl::Kernel(program.getProgram(), "phCopyAction");
}

void AgentSPG::simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng) {
	// Feed forward
	cl_int2 prevLayerSize = _layers.front()._sc.getVisibleLayerDesc(0)._size;
	cl::Image2D prevLayerState = input;

	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates(2);

			// Modulate
			{
				int argIndex = 0;

				_modulateKernel.setArg(argIndex++, prevLayerState);
				_modulateKernel.setArg(argIndex++, _layers[l]._predAttentionFeedForward.getHiddenStates()[_back]);
				_modulateKernel.setArg(argIndex++, _layers[l]._modulatedFeedForwardInput);
				_modulateKernel.setArg(argIndex++, _layerDescs[l]._minAttention);

				cs.getQueue().enqueueNDRangeKernel(_modulateKernel, cl::NullRange, cl::NDRange(prevLayerSize.x, prevLayerSize.y));
			}

			// Modulate
			{
				int argIndex = 0;

				_modulateKernel.setArg(argIndex++, _layers[l]._scHiddenStatesPrev);
				_modulateKernel.setArg(argIndex++, _layers[l]._predAttentionRecurrent.getHiddenStates()[_back]);
				_modulateKernel.setArg(argIndex++, _layers[l]._modulatedRecurrentInput);
				_modulateKernel.setArg(argIndex++, _layerDescs[l]._minAttention);

				cs.getQueue().enqueueNDRangeKernel(_modulateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y));
			}

			visibleStates[0] = _layers[l]._modulatedFeedForwardInput;
			visibleStates[1] = _layers[l]._modulatedRecurrentInput;

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);

			_layers[l]._sc.learn(cs, _layers[l]._reward, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
		}

		// Get reward
		if (l == 0) {
			int argIndex = 0;

			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._predAction.getVisibleLayer(0)._errors);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._baseLines[_back]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._baseLines[_front]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._reward);
			_baseLineUpdateKernel.setArg(argIndex++, _layerDescs[l]._baseLineDecay);
			_baseLineUpdateKernel.setArg(argIndex++, _layerDescs[l]._baseLineSensitivity);

			cs.getQueue().enqueueNDRangeKernel(_baseLineUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y));
		}
		else {
			int argIndex = 0;

			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l - 1]._predAction.getVisibleLayer(1)._errors);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l]._predAction.getVisibleLayer(0)._errors);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l]._baseLines[_back]);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l]._baseLines[_front]);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l]._reward);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layerDescs[l]._baseLineDecay);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layerDescs[l]._baseLineSensitivity);

			cs.getQueue().enqueueNDRangeKernel(_baseLineUpdateSumErrorKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y));
		}

		prevLayerState = _layers[l]._sc.getHiddenStates()[_back];
		prevLayerSize = _layerDescs[l]._hiddenSize;
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStates;

		if (l < _layers.size() - 1) {
			visibleStates.resize(2);

			visibleStates[0] = _layers[l]._sc.getHiddenStates()[_back];
			visibleStates[1] = _layers[l + 1]._predAction.getHiddenStates()[_back];
		}
		else {
			visibleStates.resize(1);

			visibleStates[0] = _layers[l]._sc.getHiddenStates()[_back];
		}

		_layers[l]._predAction.activate(cs, visibleStates, l != 0 , _layerDescs[l]._noise, rng);
		_layers[l]._predAttentionFeedForward.activate(cs, visibleStates, false, _layerDescs[l]._noise, rng);
		_layers[l]._predAttentionRecurrent.activate(cs, visibleStates, false, _layerDescs[l]._noise, rng);

		if (l == 0)
			_layers[l]._predAction.propagateError(cs, input);
		else
			_layers[l]._predAction.propagateError(cs, _layers[l - 1]._sc.getHiddenStates()[_back]);
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStatesPrev;

		if (l < _layers.size() - 1) {
			visibleStatesPrev.resize(2);

			visibleStatesPrev[0] = _layers[l]._scHiddenStatesPrev;
			visibleStatesPrev[1] = _layers[l + 1]._predAction.getHiddenStates()[_front];
		}
		else {
			visibleStatesPrev.resize(1);

			visibleStatesPrev[0] = _layers[l]._scHiddenStatesPrev;
		}

		if (l == 0) {
			_layers[l]._predAction.learnTrace(cs, reward, _layerDescs[l]._gamma, _action, visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
			_layers[l]._predAttentionFeedForward.learnTrace(cs, reward, _layerDescs[l]._gamma, _layers[l]._predAttentionFeedForward.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
			_layers[l]._predAttentionRecurrent.learnTrace(cs, reward, _layerDescs[l]._gamma, _layers[l]._predAttentionRecurrent.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
		}
		else {
			_layers[l]._predAction.learnTrace(cs, reward, _layerDescs[l]._gamma, _layers[l - 1]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
			_layers[l]._predAttentionFeedForward.learnTrace(cs, reward, _layerDescs[l]._gamma, _layers[l]._predAttentionFeedForward.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
			_layers[l]._predAttentionRecurrent.learnTrace(cs, reward, _layerDescs[l]._gamma, _layers[l]._predAttentionRecurrent.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
		}
	}

	// Copy action
	{
		int argIndex = 0;

		_copyActionKernel.setArg(argIndex++, _layers.front()._predAction.getHiddenStates()[_back]);
		_copyActionKernel.setArg(argIndex++, _action);

		cs.getQueue().enqueueNDRangeKernel(_copyActionKernel, cl::NullRange, cl::NDRange(_layers.front()._predAction.getHiddenSize().x, _layers.front()._predAction.getHiddenSize().y));
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);

		std::swap(_layers[l]._baseLines[_front], _layers[l]._baseLines[_back]);
	}
}

void AgentSPG::clearMemory(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
	}
}
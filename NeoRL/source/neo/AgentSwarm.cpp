#include "AgentSwarm.h"

using namespace neo;

void AgentSwarm::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerPredictorRadius, const std::vector<LayerDesc> &layerDescs,
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
		scDescs[0]._useTraces = false;

		scDescs[1]._size = _layerDescs[l]._hiddenSize;
		scDescs[1]._radius = _layerDescs[l]._recurrentRadius;
		scDescs[1]._ignoreMiddle = true;
		scDescs[1]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
		scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
		scDescs[1]._useTraces = false;

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._hiddenSize, _layerDescs[l]._lateralRadius, initWeightRange, initThreshold, rng);

		_layers[l]._modulatedFeedForwardInput = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerSize.x, prevLayerSize.y);

		_layers[l]._modulatedRecurrentInput = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y);

		std::vector<Predictor::VisibleLayerDesc> predDescs;
	
		if (l < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[l]._hiddenSize;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;

			predDescs[1]._size = _layerDescs[l + 1]._hiddenSize;
			predDescs[1]._radius = _layerDescs[l]._feedBackRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[l]._hiddenSize;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;
		}

		_layers[l]._pred.createRandom(cs, program, predDescs, prevLayerSize, initWeightRange, rng);

		std::vector<Swarm::VisibleLayerDesc> swarmDescs;

		if (l == 0) {
			swarmDescs.resize(3);

			swarmDescs[0]._size = inputSize;
			swarmDescs[0]._qRadius = _layerDescs[l]._qRadiusHiddenFeedForwardAttention;
			swarmDescs[0]._startRadius = _layerDescs[l]._startRadiusHiddenFeedForwardAttention;

			swarmDescs[1]._size = _layerDescs[l]._hiddenSize;
			swarmDescs[1]._qRadius = _layerDescs[l]._qRadiusHiddenRecurrentAttention;
			swarmDescs[1]._startRadius = _layerDescs[l]._startRadiusHiddenRecurrentAttention;

			swarmDescs[2]._size = actionSize;
			swarmDescs[2]._qRadius = _layerDescs[l]._qRadiusHiddenAction;
			swarmDescs[2]._startRadius = _layerDescs[l]._startRadiusHiddenAction;
		}
		else {
			swarmDescs.resize(3);

			swarmDescs[0]._size = _layerDescs[l - 1]._hiddenSize;
			swarmDescs[0]._qRadius = _layerDescs[l]._qRadiusHiddenFeedForwardAttention;
			swarmDescs[0]._startRadius = _layerDescs[l]._startRadiusHiddenFeedForwardAttention;

			swarmDescs[1]._size = _layerDescs[l]._hiddenSize;
			swarmDescs[1]._qRadius = _layerDescs[l]._qRadiusHiddenRecurrentAttention;
			swarmDescs[1]._startRadius = _layerDescs[l]._startRadiusHiddenRecurrentAttention;

			swarmDescs[2]._size = _layerDescs[l - 1]._hiddenSize;
			swarmDescs[2]._qRadius = _layerDescs[l]._qRadiusHiddenAction;
			swarmDescs[2]._startRadius = _layerDescs[l]._startRadiusHiddenAction;
		}

		_layers[l]._swarm.createRandom(cs, program, swarmDescs, _layerDescs[l]._qSize, _layerDescs[l]._hiddenSize, _layerDescs[l]._qRadius, initWeightRange, rng);
		
		// Create baselines
		_layers[l]._baseLines = createDoubleBuffer2D(cs, _layerDescs[l]._hiddenSize, CL_R, CL_FLOAT);

		_layers[l]._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y);

		_layers[l]._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

		if (l != 0) {
			cl::array<cl::size_type, 3> actionRegion = { _layers[l]._swarm.getVisibleLayerDesc(2)._size.x, _layers[l]._swarm.getVisibleLayerDesc(2)._size.y, 1 };

			_layers[l]._inhibitedAction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), swarmDescs[1]._size.x, swarmDescs[1]._size.y);

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
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._hiddenSize.x, _layerDescs.back()._hiddenSize.y, 1 };

		_lastLayerAction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs.back()._hiddenSize.x, _layerDescs.back()._hiddenSize.y);

		cs.getQueue().enqueueFillImage(_lastLayerAction, zeroColor, zeroOrigin, layerRegion);
	}

	_baseLineUpdateKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdate");
	_baseLineUpdateSumErrorKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdateSumError");
	_inhibitKernel = cl::Kernel(program.getProgram(), "phInhibit");
	_modulateKernel = cl::Kernel(program.getProgram(), "phModulate");
}

void AgentSwarm::simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng) {
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
				_modulateKernel.setArg(argIndex++, _layers[l]._swarm.getVisibleLayer(0)._actionsExploratory);
				_modulateKernel.setArg(argIndex++, _layers[l]._modulatedFeedForwardInput);
				_modulateKernel.setArg(argIndex++, _layerDescs[l]._minAttention);

				cs.getQueue().enqueueNDRangeKernel(_modulateKernel, cl::NullRange, cl::NDRange(prevLayerSize.x, prevLayerSize.y));
			}

			// Modulate
			{
				int argIndex = 0;

				_modulateKernel.setArg(argIndex++, _layers[l]._scHiddenStatesPrev);
				_modulateKernel.setArg(argIndex++, _layers[l]._swarm.getVisibleLayer(1)._actionsExploratory);
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

			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._pred.getVisibleLayer(0)._errors);
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

			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l - 1]._pred.getVisibleLayer(1)._errors);
			_baseLineUpdateSumErrorKernel.setArg(argIndex++, _layers[l]._pred.getVisibleLayer(0)._errors);
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
			visibleStates[1] = _layers[l + 1]._pred.getHiddenStates()[_back];
		}
		else {
			visibleStates.resize(1);

			visibleStates[0] = _layers[l]._sc.getHiddenStates()[_back];
		}

		_layers[l]._pred.activate(cs, visibleStates, true);

		if (l == 0)
			_layers[l]._pred.propagateError(cs, input);
		else
			_layers[l]._pred.propagateError(cs, _layers[l - 1]._sc.getHiddenStates()[_back]);
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStatesPrev;

		if (l < _layers.size() - 1) {
			visibleStatesPrev.resize(2);

			visibleStatesPrev[0] = _layers[l]._scHiddenStatesPrev;
			visibleStatesPrev[1] = _layers[l + 1]._pred.getHiddenStates()[_front];
		}
		else {
			visibleStatesPrev.resize(1);

			visibleStatesPrev[0] = _layers[l]._scHiddenStatesPrev;
		}

		if (l == 0)
			_layers[l]._pred.learn(cs, input, visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
		else
			_layers[l]._pred.learn(cs, _layers[l - 1]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
	}

	// Swarm
	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStatesPrev;

		if (l < _layers.size() - 1) {
			_layers[l]._swarm.simStep(cs, reward, _layers[l]._sc.getHiddenStates()[_back], _layers[l + 1]._inhibitedAction,
				_layerDescs[l]._swarmExpPert, _layerDescs[l]._swarmExpBreak,
				_layerDescs[l]._swarmAnnealingIterations, _layerDescs[l]._swarmActionDeriveAlpha,
				_layerDescs[l]._swarmQHiddenAlpha, _layerDescs[l]._swarmQAlpha, _layerDescs[l]._swarmPredAlpha,
				_layerDescs[l]._swarmLambda, _layerDescs[l]._swarmGamma, rng);
		}
		else {
			_layers[l]._swarm.simStep(cs, reward, _layers[l]._sc.getHiddenStates()[_back], _lastLayerAction,
				_layerDescs[l]._swarmExpPert, _layerDescs[l]._swarmExpBreak,
				_layerDescs[l]._swarmAnnealingIterations, _layerDescs[l]._swarmActionDeriveAlpha,
				_layerDescs[l]._swarmQHiddenAlpha, _layerDescs[l]._swarmQAlpha, _layerDescs[l]._swarmPredAlpha,
				_layerDescs[l]._swarmLambda, _layerDescs[l]._swarmGamma, rng);
		}

		// If not first layer, inhibit the action
		if (l != 0) {
			int argIndex = 0;

			_inhibitKernel.setArg(argIndex++, _layers[l]._swarm.getVisibleLayer(2)._actionsExploratory);
			_inhibitKernel.setArg(argIndex++, _layers[l]._inhibitedAction);
			_inhibitKernel.setArg(argIndex++, _layerDescs[l - 1]._hiddenSize);
			_inhibitKernel.setArg(argIndex++, _layerDescs[l - 1]._lateralRadius);
			_inhibitKernel.setArg(argIndex++, _layerDescs[l - 1]._scActiveRatio);

			cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_layerDescs[l - 1]._hiddenSize.x, _layerDescs[l - 1]._hiddenSize.y));
		}
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);

		std::swap(_layers[l]._baseLines[_front], _layers[l]._baseLines[_back]);
	}
}

void AgentSwarm::clearMemory(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._hiddenSize.x, _layerDescs[l]._hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
	}
}
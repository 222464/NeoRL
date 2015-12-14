#include "PredictiveHierarchy.h"

using namespace neo;

void PredictiveHierarchy::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, const std::vector<LayerDesc> &layerDescs,
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

		scDescs[1]._size = _layerDescs[l]._size;
		scDescs[1]._radius = _layerDescs[l]._recurrentRadius;
		scDescs[1]._ignoreMiddle = true;
		scDescs[1]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
		scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
		scDescs[1]._useTraces = true;

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, initThreshold, rng);

		std::vector<Predictor::VisibleLayerDesc> predDescs;

		if (l < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[l]._size;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;

			predDescs[1]._size = _layerDescs[l + 1]._size;
			predDescs[1]._radius = _layerDescs[l]._feedBackRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[l]._size;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;
		}

		_layers[l]._pred.createRandom(cs, program, predDescs, prevLayerSize, initWeightRange, false, rng);

		// Create baselines
		_layers[l]._baseLines = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);

		_layers[l]._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		_layers[l]._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._baseLines[_back], zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
	}

	_baseLineUpdateKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdate");
	_baseLineUpdateSumErrorKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdateSumError");
}

void PredictiveHierarchy::simStep(sys::ComputeSystem &cs, const cl::Image2D &input, bool learn) {
	// Feed forward
	cl::Image2D prevLayerState = input;

	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates(2);

			visibleStates[0] = prevLayerState;
			visibleStates[1] = _layers[l]._scHiddenStatesPrev;

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);

			if (learn)
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

			cs.getQueue().enqueueNDRangeKernel(_baseLineUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
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

			cs.getQueue().enqueueNDRangeKernel(_baseLineUpdateSumErrorKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
		}

		prevLayerState = _layers[l]._sc.getHiddenStates()[_back];
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

		_layers[l]._pred.activate(cs, visibleStates, l != 0);

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

		if (learn) {
			if (l == 0)
				_layers[l]._pred.learn(cs, input, visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
			else
				_layers[l]._pred.learn(cs, _layers[l - 1]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
		}
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);

		std::swap(_layers[l]._baseLines[_front], _layers[l]._baseLines[_back]);
	}
}

void PredictiveHierarchy::clearMemory(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
	}
}
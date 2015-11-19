#include "AgentCACLA.h"

#include <iostream>

using namespace neo;

void AgentCACLA::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int firstLayerPredictorRadius,
	const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
	cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange, std::mt19937 &rng)
{
	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	_inputTypes = inputTypes;

	_inputs.clear();
	_inputs.assign(_inputTypes.size(), 0.0f);

	_predictions.clear();
	_predictions.assign(_inputTypes.size(), 0.0f);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	_qInputs.clear();

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _q) {
			QInput qi;
			qi._index = i;
			qi._offset = dist01(rng) * 2.0f - 1.0f;

			_qInputs.push_back(qi);
		}
	}

	_input = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputSize.x, inputSize.y);

	cl_int2 prevLayerSize = inputSize;

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<SparseCoder::VisibleLayerDesc> scDescs(2);

		scDescs[0]._size = prevLayerSize;
		scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;

		scDescs[1]._size = _layerDescs[l]._size;
		scDescs[1]._radius = _layerDescs[l]._recurrentRadius;

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, initLateralWeightRange, initThreshold, initCodeRange, initReconstructionErrorRange, true, rng);

		std::vector<Predictor::VisibleLayerDesc> predDescs;

		if (l < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[l + 1]._size;
			predDescs[0]._radius = _layerDescs[l]._feedBackRadius;

			predDescs[1]._size = _layerDescs[l]._size;
			predDescs[1]._radius = _layerDescs[l]._predictiveRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[l]._size;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;
		}

		_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l]._size, initWeightRange, true, rng);

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

	std::vector<Predictor::VisibleLayerDesc> predDescs(1);

	predDescs[0]._size = _layerDescs.front()._size;
	predDescs[0]._radius = firstLayerPredictorRadius;

	_firstLayerPred.createRandom(cs, program, predDescs, inputSize, initWeightRange, true, rng);

	_baseLineUpdateKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdate");
}

void AgentCACLA::simStep(float reward, sys::ComputeSystem &cs, std::mt19937 &rng, bool learn) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationStdDev);
	
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> inputRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };

	// Write input
	cs.getQueue().enqueueWriteImage(_input, CL_TRUE, zeroOrigin, inputRegion, 0, 0, _inputs.data());

	// Feed forward
	cl::Image2D prevLayerState = _input;

	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates(2);

			visibleStates[0] = prevLayerState;
			visibleStates[1] = _layers[l]._scHiddenStatesPrev;

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scIterations, _layerDescs[l]._scLeak);
		}

		// Get reward
		{
			int argIndex = 0;

			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._pred.getHiddenStates()[_back]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._baseLines[_back]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._baseLines[_front]);
			_baseLineUpdateKernel.setArg(argIndex++, _layers[l]._reward);
			_baseLineUpdateKernel.setArg(argIndex++, _layerDescs[l]._baseLineDecay);
			_baseLineUpdateKernel.setArg(argIndex++, _layerDescs[l]._baseLineSensitivity);

			cs.getQueue().enqueueNDRangeKernel(_baseLineUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
		}

		prevLayerState = _layers[l]._sc.getHiddenStates()[_back];
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStates;
		
		if (l < _layers.size() - 1) {
			visibleStates.resize(2);

			visibleStates[0] = _layers[l + 1]._pred.getHiddenStates()[_back];
			visibleStates[1] = _layers[l]._sc.getHiddenStates()[_back];
		}
		else {
			visibleStates.resize(1);

			visibleStates[0] = _layers[l]._sc.getHiddenStates()[_back];
		}

		_layers[l]._pred.activate(cs, visibleStates, true);

		_layers[l]._sc.learnTrace(cs, _layers[l]._reward, _layerDescs[l]._scWeightAlpha, _layerDescs[l]._scLateralWeightAlpha, _layerDescs[l]._scWeightTraceLambda, _layerDescs[l]._scThresholdAlpha, _layerDescs[l]._scActiveRatio);
	}

	{
		std::vector<cl::Image2D> visibleStates(1);
		
		visibleStates[0] = _layers.front()._pred.getHiddenStates()[_back];
		
		_firstLayerPred.activate(cs, visibleStates, false);
	}

	// Retrieve predictions
	cs.getQueue().enqueueReadImage(_firstLayerPred.getHiddenStates()[_back], CL_TRUE, zeroOrigin, inputRegion, 0, 0, _predictions.data());

	// Gather Q
	float q = 0.0f;

	for (int i = 0; i < _qInputs.size(); i++)
		q += _predictions[_qInputs[i]._index] - _qInputs[i]._offset;

	q /= _qInputs.size();

	float tdError = reward + _gamma * q - _prevValue;

	float newQ = _prevValue + tdError * _qAlpha;
	std::cout << newQ << std::endl;
	_prevValue = q;

	// Learn predictions modulated by reward
	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStatesPrev;

		if (l < _layers.size() - 1) {
			visibleStatesPrev.resize(2);

			visibleStatesPrev[0] = _layers[l + 1]._pred.getHiddenStates()[_front];
			visibleStatesPrev[1] = _layers[l]._scHiddenStatesPrev;
		}
		else {
			visibleStatesPrev.resize(1);

			visibleStatesPrev[0] = _layers[l]._scHiddenStatesPrev;
		}

		_layers[l]._pred.learnTrace(cs, tdError, _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
	}

	{
		std::vector<cl::Image2D> visibleStatesPrev(1);

		visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

		_firstLayerPred.learnTrace(cs, tdError, _input, visibleStatesPrev, _predWeightAlpha, _predWeightLambda);
	}

	// Alter inputs
	for (int i = 0; i < _inputTypes.size(); i++) {
		switch (_inputTypes[i]) {
		case _action:
			if (dist01(rng) < _explorationBreakChance)
				_inputs[i] = dist01(rng) * 2.0f - 1.0f;
			else
				_inputs[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _predictions[i])) + pertDist(rng)));

			break;
		}
	}

	for (int i = 0; i < _qInputs.size(); i++)
		_inputs[_qInputs[i]._index] = _prevValue + _qInputs[i]._offset + 1.0f;

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);
	}
}
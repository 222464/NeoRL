#include "AgentQRoute.h"

#include <algorithm>

#include <iostream>

using namespace neo;

void AgentQRoute::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int firstLayerPredictorRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
	cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange,
	std::mt19937 &rng)
{
	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());
	_inputTypes = inputTypes;

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _action)
			_actionIndices.push_back(i);
		else if (_inputTypes[i] == _antiAction)
			_antiActionIndices.push_back(i);
	}

	assert(_actionIndices.size() == _antiActionIndices.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
	cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

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

		_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l]._size, initWeightRange, false, rng);

		// Create baselines
		_layers[l]._baseLines = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);

		_layers[l]._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);
		_layers[l]._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._baseLines[_back], zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);

		// Q
		int qWeightDiam = _layerDescs[l]._qRadius * 2 + 1;
		int numQWeights = qWeightDiam * qWeightDiam;

		_layers[l]._qStates = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);

		cs.getQueue().enqueueFillImage(_layers[l]._qStates[_back], zeroColor, zeroOrigin, layerRegion);

		cl_int3 qWeightsSize = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, numQWeights };

		_layers[l]._qWeights = createDoubleBuffer3D(cs, qWeightsSize, CL_RG, CL_FLOAT);

		randomUniform(_layers[l]._qWeights[_back], cs, randomUniform3DKernel, qWeightsSize, initWeightRange, rng);

		_layers[l]._qBiases = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_RG, CL_FLOAT);

		randomUniform(_layers[l]._qBiases[_back], cs, randomUniform2DKernel, _layerDescs[l]._size, initWeightRange, rng);

		_layers[l]._qErrorTemp = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);
	}

	std::vector<Predictor::VisibleLayerDesc> predDescs(1);

	predDescs[0]._size = _layerDescs.front()._size;
	predDescs[0]._radius = firstLayerPredictorRadius;

	_firstLayerPred.createRandom(cs, program, predDescs, inputSize, initWeightRange, false, rng);

	_baseLineUpdateKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdate");

	_qForwardKernel = cl::Kernel(program.getProgram(), "qForward");
	_qBackwardKernel = cl::Kernel(program.getProgram(), "qBackward");
	_qBackwardFirstLayerKernel = cl::Kernel(program.getProgram(), "qBackwardFirstLayer");
	_qWeightUpdateKernel = cl::Kernel(program.getProgram(), "qWeightUpdate");

	_qConnections.resize(_layerDescs.back()._size.x * _layerDescs.back()._size.y);

	std::uniform_real_distribution<float> initWeightDist(initWeightRange.x, initWeightRange.y);

	for (int i = 0; i < _qConnections.size(); i++)
		_qConnections[i]._weight = initWeightDist(rng);

	_qStates.resize(_qConnections.size());
	_qErrors.resize(_qConnections.size());
	_scStates.resize(_qConnections.size());

	_qInputLayerErrors.resize(inputSize.x * inputSize.y);
	_inputLayerStates.clear();
	_inputLayerStates.assign(_qInputLayerErrors.size(), 0.0f);
	_prediction.resize(_qInputLayerErrors.size());

	_lastLayerError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs.back()._size.x, _layerDescs.back()._size.y);
	_inputLayerError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputSize.x, inputSize.y);

	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { inputSize.x, inputSize.y, 1 };
		
		_input = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputSize.x, inputSize.y);

		cs.getQueue().enqueueFillImage(_input, zeroColor, zeroOrigin, layerRegion);
	}
}

void AgentQRoute::simStep(float reward, sys::ComputeSystem &cs, std::mt19937 &rng, bool learn) {
	// Feed forward
	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };

		cs.getQueue().enqueueWriteImage(_input, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _inputLayerStates.data());
	}

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

		_layers[l]._pred.learn(cs, _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
	}

	{
		std::vector<cl::Image2D> visibleStatesPrev(1);

		visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

		_firstLayerPred.learn(cs, _input, visibleStatesPrev, _predWeightAlpha);
	}

	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };

		cs.getQueue().enqueueReadImage(_firstLayerPred.getHiddenStates()[_back], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _prediction.data());
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationPerturbationStdDev);

	// Set predicted action as starting point
	_inputLayerStates = _prediction;

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _action)
			_inputLayerStates[i] = std::min(1.0f, std::max(-1.0f, _prediction[i]));
	}

	// Write initial inputs
	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };

		cs.getQueue().enqueueWriteImage(_input, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _inputLayerStates.data());
	}

	// Optimize actions to maximize Q
	float q;

	for (int it = 0; it < _qIter; it++) {
		// Forwards
		prevLayerState = _input;

		cl_int2 prevLayerSize = _firstLayerPred.getHiddenSize();

		for (int l = 0; l < _layers.size(); l++) {
			int argIndex = 0;
			
			_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_qForwardKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
			_qForwardKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
			_qForwardKernel.setArg(argIndex++, prevLayerState);
			_qForwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
			_qForwardKernel.setArg(argIndex++, prevLayerSize);
			_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getVisibleLayer(0)._hiddenToVisible);
			_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
			_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

			prevLayerState = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}

		// Compute Q
		{
			cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._size.x, _layerDescs.back()._size.y, 1 };

			cs.getQueue().enqueueReadImage(_layers.back()._qStates[_front], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qStates.data());

			q = 0.0f;

			for (int i = 0; i < _qStates.size(); i++)
				q += _qStates[i] * _qConnections[i]._weight;

			//q /= _qStates.size();

			std::cout << "Q: " << q << std::endl;
		}

		// Backwards
		{
			cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._size.x, _layerDescs.back()._size.y, 1 };

			// Last layer error
			cs.getQueue().enqueueReadImage(_layers.back()._sc.getHiddenStates()[_back], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _scStates.data());

			cs.getQueue().enqueueReadImage(_layers.back()._qStates[_front], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qStates.data());
			
			for (int i = 0; i < _qErrors.size(); i++)
				_qErrors[i] = _scStates[i] * (_qStates[i] > 0.0f && _qStates[i] < 1.0f ? 1.0f : _layerDescs.back()._qReluLeak) * _qConnections[i]._weight;

			//for (int i = 0; i < _qErrors.size(); i++)
			//	_qErrors[i] = _qStates[i] * (1.0f - _qStates[i]) * _qConnections[i]._weight;

			cs.getQueue().enqueueWriteImage(_layers.back()._qErrorTemp, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qErrors.data());
		}

		for (int l = _layers.size() - 2; l >= 0; l--) {
			int argIndex = 0;

			cl_int2 reverseRadii = { static_cast<int>(std::ceil(_layers[l + 1]._sc.getVisibleLayer(0)._visibleToHidden.x * _layerDescs[l + 1]._qRadius)),
				static_cast<int>(std::ceil(_layers[l + 1]._sc.getVisibleLayer(0)._visibleToHidden.y * _layerDescs[l + 1]._qRadius)) };

			_qBackwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._qWeights[_back]);
			_qBackwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
			_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._qErrorTemp);
			_qBackwardKernel.setArg(argIndex++, _layers[l]._qErrorTemp); 
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._size);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l + 1]._size);
			_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._sc.getVisibleLayer(0)._visibleToHidden);
			_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._sc.getVisibleLayer(0)._hiddenToVisible);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l + 1]._qRadius);
			_qBackwardKernel.setArg(argIndex++, reverseRadii);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qBackwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
		}

		// First layer
		{
			int argIndex = 0;

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(_layers.front()._sc.getVisibleLayer(0)._visibleToHidden.x * _layerDescs.front()._qRadius)),
				static_cast<int>(std::ceil(_layers.front()._sc.getVisibleLayer(0)._visibleToHidden.y * _layerDescs.front()._qRadius)) };

			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._qWeights[_back]);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._qErrorTemp);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _inputLayerError);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayerDesc(0)._size);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layerDescs.front()._size);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayer(0)._visibleToHidden);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayer(0)._hiddenToVisible);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layerDescs.front()._qRadius);
			_qBackwardFirstLayerKernel.setArg(argIndex++, reverseRadii);

			cs.getQueue().enqueueNDRangeKernel(_qBackwardFirstLayerKernel, cl::NullRange, cl::NDRange(_layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y));
		}

		{
			cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> layerRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };
			
			cs.getQueue().enqueueReadImage(_inputLayerError, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qInputLayerErrors.data());
		}

		// Move actions - final iteration has exploration
		if (it == _qIter - 1) {	
			for (int i = 0; i < _actionIndices.size(); i++) {
				if (dist01(rng) < _explorationBreakChance)
					_inputLayerStates[_actionIndices[i]] = dist01(rng);
				else
					_inputLayerStates[_actionIndices[i]] = std::min(1.0f, std::max(-1.0f, _inputLayerStates[_actionIndices[i]] + pertDist(rng) + _actionDeriveAlpha * ((_qInputLayerErrors[_actionIndices[i]] - _qInputLayerErrors[_antiActionIndices[i]]) > 0.0f ? 1.0f : -1.0f)));
			}
		}
		else {
			for (int i = 0; i < _actionIndices.size(); i++)
				_inputLayerStates[_actionIndices[i]] = std::min(1.0f, std::max(-1.0f, _inputLayerStates[_actionIndices[i]] + _actionDeriveAlpha * ((_qInputLayerErrors[_actionIndices[i]] - _qInputLayerErrors[_antiActionIndices[i]]) > 0.0f ? 1.0f : -1.0f)));
		}

		// Set anti-actions
		for (int i = 0; i < _antiActionIndices.size(); i++)
			_inputLayerStates[_antiActionIndices[i]] = -_inputLayerStates[_actionIndices[i]];

		// Write new annealed inputs
		{
			cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> layerRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };

			cs.getQueue().enqueueWriteImage(_input, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _inputLayerStates.data());
		}
	}

	// Last forwards
	prevLayerState = _input;

	cl_int2 prevLayerSize = _firstLayerPred.getHiddenSize();

	for (int l = 0; l < _layers.size(); l++) {
		int argIndex = 0;

		_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
		_qForwardKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
		_qForwardKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
		_qForwardKernel.setArg(argIndex++, prevLayerState);
		_qForwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
		_qForwardKernel.setArg(argIndex++, prevLayerSize);
		_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getVisibleLayer(0)._hiddenToVisible);
		_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
		_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

		cs.getQueue().enqueueNDRangeKernel(_qForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

		prevLayerState = _layers[l]._qStates[_front];
		prevLayerSize = _layerDescs[l]._size;
	}

	float maxQ = q;

	// Compute Q
	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._size.x, _layerDescs.back()._size.y, 1 };

		cs.getQueue().enqueueReadImage(_layers.back()._qStates[_front], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qStates.data());

		q = 0.0f;

		for (int i = 0; i < _qStates.size(); i++)
			q += _qStates[i] * _qConnections[i]._weight;

		//q /= _qStates.size();

		//std::cout << "Q: " << q << std::endl;
	}

	// Last backwards (for gradient update)
	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._size.x, _layerDescs.back()._size.y, 1 };

		// Last layer error
		cs.getQueue().enqueueReadImage(_layers.back()._sc.getHiddenStates()[_back], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _scStates.data());

		cs.getQueue().enqueueReadImage(_layers.back()._qStates[_front], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qStates.data());

		for (int i = 0; i < _qErrors.size(); i++)
			_qErrors[i] = _scStates[i] * (_qStates[i] > 0.0f && _qStates[i] < 1.0f ? 1.0f : _layerDescs.back()._qReluLeak) * _qConnections[i]._weight;

		//for (int i = 0; i < _qErrors.size(); i++)
		//	_qErrors[i] = _qStates[i] * (1.0f - _qStates[i]) * _qConnections[i]._weight;

		cs.getQueue().enqueueWriteImage(_layers.back()._qErrorTemp, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qErrors.data());
	}

	for (int l = _layers.size() - 2; l >= 0; l--) {
		int argIndex = 0;

		cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(_layers[l + 1]._sc.getVisibleLayer(0)._visibleToHidden.x * _layerDescs[l + 1]._qRadius)),
			static_cast<int>(std::ceil(_layers[l + 1]._sc.getVisibleLayer(0)._visibleToHidden.y * _layerDescs[l + 1]._qRadius)) };

		_qBackwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
		_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._qWeights[_back]);
		_qBackwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
		_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._qErrorTemp);
		_qBackwardKernel.setArg(argIndex++, _layers[l]._qErrorTemp);
		_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._size);
		_qBackwardKernel.setArg(argIndex++, _layerDescs[l + 1]._size);
		_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._sc.getVisibleLayer(0)._visibleToHidden);
		_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._sc.getVisibleLayer(0)._hiddenToVisible);
		_qBackwardKernel.setArg(argIndex++, _layerDescs[l + 1]._qRadius);
		_qBackwardKernel.setArg(argIndex++, reverseRadii);
		_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

		cs.getQueue().enqueueNDRangeKernel(_qBackwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
	}

	// First layer
	{
		int argIndex = 0;

		cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(_layers.front()._sc.getVisibleLayer(0)._visibleToHidden.x * _layerDescs.front()._qRadius)),
			static_cast<int>(std::ceil(_layers.front()._sc.getVisibleLayer(0)._visibleToHidden.y * _layerDescs.front()._qRadius)) };

		_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._qWeights[_back]);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._qErrorTemp);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _inputLayerError);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayerDesc(0)._size);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _layerDescs.front()._size);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayer(0)._visibleToHidden);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayer(0)._hiddenToVisible);
		_qBackwardFirstLayerKernel.setArg(argIndex++, _layerDescs.front()._qRadius);
		_qBackwardFirstLayerKernel.setArg(argIndex++, reverseRadii);

		cs.getQueue().enqueueNDRangeKernel(_qBackwardFirstLayerKernel, cl::NullRange, cl::NDRange(_layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y));
	}

	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };

		cs.getQueue().enqueueReadImage(_inputLayerError, CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qInputLayerErrors.data());
	}

	// Q
	float tdError = reward + _gamma * q - _prevValue;

	for (int i = 0; i < _qConnections.size(); i++) {
		_qConnections[i]._weight += _lastLayerQAlpha * tdError * _qConnections[i]._trace;

		_qConnections[i]._trace = _lastLayerQGammaLambda * _qConnections[i]._trace + _qStates[i];
	}

	// Weight update
	prevLayerState = _input;

	prevLayerSize = _firstLayerPred.getHiddenSize();

	if (learn) {
		for (int l = 0; l < _layers.size(); l++) {
			int argIndex = 0;

			_qWeightUpdateKernel.setArg(argIndex++, prevLayerState);
			_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
			_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qErrorTemp);
			_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
			_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qWeights[_front]);
			_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
			_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qBiases[_front]);
			_qWeightUpdateKernel.setArg(argIndex++, prevLayerSize);
			_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._size);
			_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
			_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qAlpha);
			_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qGammaLambda);
			_qWeightUpdateKernel.setArg(argIndex++, tdError);

			cs.getQueue().enqueueNDRangeKernel(_qWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

			prevLayerState = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);
	
		std::swap(_layers[l]._qStates[_front], _layers[l]._qStates[_back]);

		if (learn) {
			std::swap(_layers[l]._qWeights[_front], _layers[l]._qWeights[_back]);

			std::swap(_layers[l]._qBiases[_front], _layers[l]._qBiases[_back]);
		}

		std::swap(_layers[l]._baseLines[_front], _layers[l]._baseLines[_back]);
	}

	_prevValue = q;
}
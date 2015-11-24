#include "AgentQRoute.h"

#include <algorithm>

#include <iostream>

using namespace neo;

void AgentQRoute::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int inputPredictorRadius, cl_int actionPredictorRadius,
	cl_int actionFeedForwardRadius, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
	cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange, std::mt19937 &rng)
{
	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());
	
	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
	cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

	_inputsImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputSize.x, inputSize.y);
	_actionsImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), actionSize.x, actionSize.y);
	_actionsExploratoryImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), actionSize.x, actionSize.y);

	for (int l = 0; l < _layers.size(); l++) {
		if (l == 0) {
			std::vector<SparseCoder::VisibleLayerDesc> scDescs(2);

			scDescs[0]._size = inputSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;

			scDescs[1]._size = actionSize;
			scDescs[1]._radius = actionFeedForwardRadius;

			//scDescs[2]._size = _layerDescs[l]._size;
			//scDescs[2]._radius = _layerDescs[l]._recurrentRadius;

			_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, initLateralWeightRange, initThreshold, initCodeRange, initReconstructionErrorRange, true, rng);
		}
		else {
			std::vector<SparseCoder::VisibleLayerDesc> scDescs(2);

			scDescs[0]._size = _layerDescs[l - 1]._size;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;

			scDescs[1]._size = _layerDescs[l]._size;
			scDescs[1]._radius = _layerDescs[l]._recurrentRadius;

			_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, initLateralWeightRange, initThreshold, initCodeRange, initReconstructionErrorRange, true, rng);
		}


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

	// Input
	{
		std::vector<Predictor::VisibleLayerDesc> predDescs(1);

		predDescs[0]._size = _layerDescs.front()._size;
		predDescs[0]._radius = inputPredictorRadius;

		_inputPred.createRandom(cs, program, predDescs, inputSize, initWeightRange, false, rng);
	}

	// Action
	{
		std::vector<Predictor::VisibleLayerDesc> predDescs(1);

		predDescs[0]._size = _layerDescs.front()._size;
		predDescs[0]._radius = actionPredictorRadius;

		_actionPred.createRandom(cs, program, predDescs, actionSize, initWeightRange, false, rng);
	}

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

	_inputs.clear();
	_inputs.assign(inputSize.x * inputSize.y, 0.0f);

	_actions.clear();
	_actions.assign(actionSize.x * actionSize.y, 0.0f);

	_actionsExploratory.clear();
	_actionsExploratory.assign(_actions.size(), 0.0f);

	_actionErrors.clear();
	_actionErrors.assign(_actions.size(), 0.0f);

	_inputPredictions.resize(_inputs.size());
	_actionPredictions.resize(_actions.size());

	_lastLayerError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs.back()._size.x, _layerDescs.back()._size.y);
	_inputLayerError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), actionSize.x, actionSize.y);
}

void AgentQRoute::simStep(float reward, sys::ComputeSystem &cs, std::mt19937 &rng, bool learn) {
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> inputRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };
	cl::array<cl::size_type, 3> actionRegion = { _layers.front()._sc.getVisibleLayerDesc(1)._size.x, _layers.front()._sc.getVisibleLayerDesc(1)._size.y, 1 };

	// Write input
	cs.getQueue().enqueueWriteImage(_inputsImage, CL_TRUE, zeroOrigin, inputRegion, 0, 0, _inputs.data());
	cs.getQueue().enqueueWriteImage(_actionsImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actions.data());
	cs.getQueue().enqueueWriteImage(_actionsExploratoryImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actionsExploratory.data());

	for (int l = 0; l < _layers.size(); l++) {
		if (l == 0) {
			std::vector<cl::Image2D> visibleStates(2);

			visibleStates[0] = _inputsImage;
			visibleStates[1] = _actionsExploratoryImage;
			//visibleStates[2] = _layers[l]._scHiddenStatesPrev;

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scIterations, _layerDescs[l]._scLeak);
		}
		else {
			std::vector<cl::Image2D> visibleStates(2);

			visibleStates[0] = _layers[l - 1]._sc.getHiddenStates()[_back];
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

	// Input
	{
		std::vector<cl::Image2D> visibleStates(1);

		visibleStates[0] = _layers.front()._pred.getHiddenStates()[_back];

		_inputPred.activate(cs, visibleStates, false);
	}

	// Action
	{
		std::vector<cl::Image2D> visibleStates(1);

		visibleStates[0] = _layers.front()._pred.getHiddenStates()[_back];

		_actionPred.activate(cs, visibleStates, false);
	}

	// Retrieve predictions
	cs.getQueue().enqueueReadImage(_inputPred.getHiddenStates()[_back], CL_TRUE, zeroOrigin, inputRegion, 0, 0, _inputPredictions.data());
	cs.getQueue().enqueueReadImage(_actionPred.getHiddenStates()[_back], CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actionPredictions.data());

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

	// Input
	{
		std::vector<cl::Image2D> visibleStatesPrev(1);

		visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

		_inputPred.learn(cs, _inputsImage, visibleStatesPrev, _predWeightAlpha);
	}

	// Action
	{
		std::vector<cl::Image2D> visibleStatesPrev(1);

		visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

		_actionPred.learn(cs, _actionsImage, visibleStatesPrev, _predWeightAlpha);
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationPerturbationStdDev);

	// Set predicted action as starting point
	_actions = _actionPredictions;

	for (int i = 0; i < _actions.size(); i++)
		_actions[i] = std::min(1.0f, std::max(-1.0f, _actions[i]));

	// Write initial inputs
	cs.getQueue().enqueueWriteImage(_actionsImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actions.data());

	// Optimize actions to maximize Q
	float q;

	for (int it = 0; it < _qIter; it++) {
		// Forwards
		cl::Image2D prevLayerState = _actionsImage;

		cl_int2 prevLayerSize = _actionPred.getHiddenSize();

		for (int l = 0; l < _layers.size(); l++) {
			int argIndex = 0;
			
			_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_qForwardKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
			_qForwardKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
			_qForwardKernel.setArg(argIndex++, prevLayerState);
			_qForwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
			_qForwardKernel.setArg(argIndex++, prevLayerSize);
			_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getVisibleLayer(l == 0 ? 1 : 0)._hiddenToVisible);
			_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
			_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

			prevLayerState = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}

		// Compute Q
		{
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
			cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._size.x, _layerDescs.back()._size.y, 1 };

			// Last layer error
			cs.getQueue().enqueueReadImage(_layers.back()._sc.getHiddenStates()[_back], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _scStates.data());

			cs.getQueue().enqueueReadImage(_layers.back()._qStates[_front], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qStates.data());
			
			//for (int i = 0; i < _qErrors.size(); i++)
			//	_qErrors[i] = _scStates[i] * (_qStates[i] > 0.0f && _qStates[i] < 1.0f ? 1.0f : _layerDescs.back()._qReluLeak) * _qConnections[i]._weight;

			for (int i = 0; i < _qErrors.size(); i++)
				_qErrors[i] = _qStates[i] * (1.0f - _qStates[i]) * _qConnections[i]._weight;

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

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(_layers.front()._sc.getVisibleLayer(1)._visibleToHidden.x * _layerDescs.front()._qRadius)),
				static_cast<int>(std::ceil(_layers.front()._sc.getVisibleLayer(1)._visibleToHidden.y * _layerDescs.front()._qRadius)) };

			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._qWeights[_back]);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._qErrorTemp);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _inputLayerError);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayerDesc(1)._size);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layerDescs.front()._size);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayer(1)._visibleToHidden);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layers.front()._sc.getVisibleLayer(1)._hiddenToVisible);
			_qBackwardFirstLayerKernel.setArg(argIndex++, _layerDescs.front()._qRadius);
			_qBackwardFirstLayerKernel.setArg(argIndex++, reverseRadii);

			cs.getQueue().enqueueNDRangeKernel(_qBackwardFirstLayerKernel, cl::NullRange, cl::NDRange(_layers.front()._sc.getVisibleLayerDesc(1)._size.x, _layers.front()._sc.getVisibleLayerDesc(1)._size.y));
		}

		cs.getQueue().enqueueReadImage(_inputLayerError, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actionErrors.data());

		// Move actions - final iteration has exploration
		if (it == _qIter - 1) {	
			for (int i = 0; i < _actions.size(); i++)
				_actions[i] = std::min(1.0f, std::max(-1.0f, _actions[i] + _actionDeriveAlpha * (_actionErrors[i] > 0.0f ? 1.0f : -1.0f)));
			std::cout << _actionErrors[0] << std::endl;
			// Write new annealed actions
			cs.getQueue().enqueueWriteImage(_actionsImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actions.data());

			for (int i = 0; i < _actions.size(); i++) {
				if (dist01(rng) < _explorationBreakChance)
					_actionsExploratory[i] = dist01(rng) * 2.0f - 1.0f;
				else
					_actionsExploratory[i] = std::min(1.0f, std::max(-1.0f, _actions[i] + pertDist(rng)));
			}

			// Write new annealed exploratory actions
			cs.getQueue().enqueueWriteImage(_actionsExploratoryImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actionsExploratory.data());
		}
		else {
			for (int i = 0; i < _actions.size(); i++)
				_actions[i] = std::min(1.0f, std::max(-1.0f, _actions[i] + _actionDeriveAlpha * (_actionErrors[i] > 0.0f ? 1.0f : -1.0f)));
		
			// Write new annealed actions
			cs.getQueue().enqueueWriteImage(_actionsImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actions.data());
		}
	}

	// Last forwards
	cl::Image2D prevLayerState = _actionsExploratoryImage;

	cl_int2 prevLayerSize = _actionPred.getHiddenSize();

	for (int l = 0; l < _layers.size(); l++) {
		int argIndex = 0;

		_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
		_qForwardKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
		_qForwardKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
		_qForwardKernel.setArg(argIndex++, prevLayerState);
		_qForwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
		_qForwardKernel.setArg(argIndex++, prevLayerSize);
		_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getVisibleLayer(l == 0 ? 1 : 0)._hiddenToVisible);
		_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
		_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

		cs.getQueue().enqueueNDRangeKernel(_qForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

		prevLayerState = _layers[l]._qStates[_front];
		prevLayerSize = _layerDescs[l]._size;
	}

	float maxQ = q;

	// Compute Q
	{
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
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs.back()._size.x, _layerDescs.back()._size.y, 1 };

		// Last layer error
		cs.getQueue().enqueueReadImage(_layers.back()._sc.getHiddenStates()[_back], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _scStates.data());

		cs.getQueue().enqueueReadImage(_layers.back()._qStates[_front], CL_TRUE, zeroOrigin, layerRegion, 0, 0, _qStates.data());

		//for (int i = 0; i < _qErrors.size(); i++)
		//	_qErrors[i] = _scStates[i] * (_qStates[i] > 0.0f && _qStates[i] < 1.0f ? 1.0f : _layerDescs.back()._qReluLeak) * _qConnections[i]._weight;

		for (int i = 0; i < _qErrors.size(); i++)
			_qErrors[i] = _qStates[i] * (1.0f - _qStates[i]) * _qConnections[i]._weight;

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

	// Q
	float tdError = reward + _gamma * q - _prevValue;

	for (int i = 0; i < _qConnections.size(); i++) {
		_qConnections[i]._weight += tdError * _qConnections[i]._trace;

		_qConnections[i]._trace = _lastLayerQGammaLambda * _qConnections[i]._trace + _lastLayerQAlpha * _qStates[i];
	}

	// Weight update
	prevLayerState = _actionsExploratoryImage;

	prevLayerSize = _actionPred.getHiddenSize();

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
			_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qBiasAlpha);
			_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qGammaLambda);
			_qWeightUpdateKernel.setArg(argIndex++, tdError);

			cs.getQueue().enqueueNDRangeKernel(_qWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

			prevLayerState = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
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
#include "AgentSwarm.h"

#include <iostream>

using namespace neo;

void AgentSwarm::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int inputPredictorRadius, cl_int actionPredictorRadius,
	cl_int actionFeedForwardRadius, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
	cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange, std::mt19937 &rng)
{
	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	_inputs.clear();
	_inputs.assign(inputSize.x * inputSize.y, 0.0f);

	_actions.clear();
	_actions.assign(actionSize.x * actionSize.y, 0.0f);

	_inputPredictions.clear();
	_inputPredictions.assign(_inputs.size(), 0.0f);

	_actionPredictions.clear();
	_actionPredictions.assign(_actions.size(), { 0.0f, 0.0f });

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	_inputsImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputSize.x, inputSize.y);
	_actionsImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), actionSize.x, actionSize.y);

	for (int l = 0; l < _layers.size(); l++) {
		if (l == 0) {
			std::vector<SparseCoder::VisibleLayerDesc> scDescs(3);

			scDescs[0]._size = inputSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;

			scDescs[1]._size = actionSize;
			scDescs[1]._radius = actionFeedForwardRadius;

			scDescs[2]._size = _layerDescs[l]._size;
			scDescs[2]._radius = _layerDescs[l]._recurrentRadius;

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

		std::vector<PredictorSwarm::VisibleLayerDesc> predDescs;

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

		_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l]._size, initWeightRange, rng);

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

	// Input
	{
		std::vector<Predictor::VisibleLayerDesc> predDescs(1);

		predDescs[0]._size = _layerDescs.front()._size;
		predDescs[0]._radius = inputPredictorRadius;

		_inputPred.createRandom(cs, program, predDescs, inputSize, initWeightRange, false, rng);
	}

	// Action
	{
		std::vector<PredictorSwarm::VisibleLayerDesc> predDescs(1);

		predDescs[0]._size = _layerDescs.front()._size;
		predDescs[0]._radius = actionPredictorRadius;

		_actionPred.createRandom(cs, program, predDescs, inputSize, initWeightRange, rng);
	}

	_baseLineUpdateKernel = cl::Kernel(program.getProgram(), "phBaseLineUpdate");
}

void AgentSwarm::simStep(float reward, sys::ComputeSystem &cs, std::mt19937 &rng, bool learn) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationStdDev);

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> inputRegion = { _layers.front()._sc.getVisibleLayerDesc(0)._size.x, _layers.front()._sc.getVisibleLayerDesc(0)._size.y, 1 };
	cl::array<cl::size_type, 3> actionRegion = { _layers.front()._sc.getVisibleLayerDesc(1)._size.x, _layers.front()._sc.getVisibleLayerDesc(1)._size.y, 1 };

	// Write input
	cs.getQueue().enqueueWriteImage(_inputsImage, CL_TRUE, zeroOrigin, inputRegion, 0, 0, _inputs.data());
	cs.getQueue().enqueueWriteImage(_actionsImage, CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actions.data());

	// Feed forward
	for (int l = 0; l < _layers.size(); l++) {
		if (l == 0) {
			std::vector<cl::Image2D> visibleStates(3);

			visibleStates[0] = _inputsImage;
			visibleStates[1] = _actionsImage;
			visibleStates[2] = _layers[l]._scHiddenStatesPrev;

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

		_actionPred.activate(cs, visibleStates, true);
	}

	// Retrieve predictions
	cs.getQueue().enqueueReadImage(_inputPred.getHiddenStates()[_back], CL_TRUE, zeroOrigin, inputRegion, 0, 0, _inputPredictions.data());
	cs.getQueue().enqueueReadImage(_actionPred.getHiddenStates()[_back], CL_TRUE, zeroOrigin, actionRegion, 0, 0, _actionPredictions.data());

	std::cout << "Q: " << _actionPredictions[0].y << std::endl;

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

		_layers[l]._pred.learnTrace(cs, reward, _layerDescs[l]._gamma, _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
	}

	// Input
	{
		std::vector<cl::Image2D> visibleStatesPrev(1);

		visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

		_inputPred.learn(cs, _inputsImage, visibleStatesPrev, _predWeightAlpha.x);
	}

	// Action
	{
		std::vector<cl::Image2D> visibleStatesPrev(1);

		visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

		_actionPred.learnTrace(cs, reward, _gamma, _actionsImage, visibleStatesPrev, _predWeightAlpha, _predWeightLambda);
	}

	// Alter inputs
	for (int i = 0; i < _actions.size(); i++) {
		if (dist01(rng) < _explorationBreakChance)
			_actions[i] = dist01(rng) * 2.0f - 1.0f;
		else
			_actions[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, _actionPredictions[i].x)) + pertDist(rng)));
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);

		std::swap(_layers[l]._baseLines[_front], _layers[l]._baseLines[_back]);
	}
}
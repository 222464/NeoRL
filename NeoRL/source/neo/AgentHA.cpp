#include "AgentHA.h"

#include <iostream>

using namespace neo;

void AgentHA::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerFeedBackRadius, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_inputSize = inputSize;
	_actionSize = actionSize;

	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
	cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs;

		if (l != 0) {
			scDescs.resize(2);

			scDescs[0]._size = _layerDescs[l - 1]._size;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[0]._useTraces = true;

			scDescs[1]._size = _layerDescs[l]._size;
			scDescs[1]._radius = _layerDescs[l]._recurrentRadius;
			scDescs[1]._ignoreMiddle = true;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
			scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[1]._useTraces = true;
		}
		else {
			scDescs.resize(3);

			scDescs[0]._size = _inputSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[0]._useTraces = true;

			scDescs[1]._size = _actionSize;
			scDescs[1]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[1]._ignoreMiddle = false;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[1]._useTraces = true;

			scDescs[2]._size = _layerDescs[l]._size;
			scDescs[2]._radius = _layerDescs[l]._recurrentRadius;
			scDescs[2]._ignoreMiddle = true;
			scDescs[2]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
			scDescs[2]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[2]._useTraces = true;
		}

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, rng);

		// Predictor
		{
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

#ifdef USE_DETERMINISTIC_POLICY_GRADIENT
			_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l]._size, initWeightRange, false, rng);
#else
			_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l]._size, initWeightRange, true, rng);
#endif
		}

		// Q
		{
			int qDiam = 2 * _layerDescs[l]._qRadius + 1;

			int numWeights = qDiam * qDiam;

			cl_int3 qWeightsSize = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, numWeights };

			_layers[l]._qWeights = createDoubleBuffer3D(cs, qWeightsSize, CL_RG, CL_FLOAT);

			randomUniform(_layers[l]._qWeights[_back], cs, randomUniform3DKernel, qWeightsSize, initWeightRange, rng);

			_layers[l]._qBiases = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);

			randomUniform(_layers[l]._qBiases[_back], cs, randomUniform2DKernel, _layerDescs[l]._size, initWeightRange, rng);

			_layers[l]._qStates = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);

			cs.getQueue().enqueueFillImage(_layers[l]._qStates[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l]._size.x), static_cast<cl::size_type>(_layerDescs[l]._size.y), 1 });
		}

		// Create baselines
		_layers[l]._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._reward, zeroColor, zeroOrigin, layerRegion);

		_layers[l]._qErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);
	}

	// Action predictor
	{
		std::vector<Predictor::VisibleLayerDesc> predDescs(1);

		predDescs[0]._size = _layerDescs.front()._size;
		predDescs[0]._radius = firstLayerFeedBackRadius;

#ifdef USE_DETERMINISTIC_POLICY_GRADIENT
		_actionPred.createRandom(cs, program, predDescs, _actionSize, initWeightRange, false, rng);
#else
		_actionPred.createRandom(cs, program, predDescs, _actionSize, initWeightRange, true, rng);
#endif
	}

	// Last Q
	{
		int qDiam = 2 * _qLastRadius + 1;

		int numWeights = qDiam * qDiam;

		cl_int3 qWeightsSize = { _qLastSize.x, _qLastSize.y, numWeights };

		_qLastWeights = createDoubleBuffer3D(cs, qWeightsSize, CL_RG, CL_FLOAT);

		randomUniform(_qLastWeights[_back], cs, randomUniform3DKernel, qWeightsSize, initWeightRange, rng);

		_qLastBiases = createDoubleBuffer2D(cs, _qLastSize, CL_R, CL_FLOAT);

		randomUniform(_qLastBiases[_back], cs, randomUniform2DKernel, _qLastSize, initWeightRange, rng);

		_qLastStates = createDoubleBuffer2D(cs, _qLastSize, CL_R, CL_FLOAT);

		cs.getQueue().enqueueFillImage(_qLastStates[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_qLastSize.x), static_cast<cl::size_type>(_qLastSize.y), 1 });
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");

	_qForwardKernel = cl::Kernel(program.getProgram(), "qForward");
	_qLastForwardKernel = cl::Kernel(program.getProgram(), "qLastForward");
	_qBackwardKernel = cl::Kernel(program.getProgram(), "qBackward");
	_qLastBackwardKernel = cl::Kernel(program.getProgram(), "qLastBackward");
	_qFirstBackwardKernel = cl::Kernel(program.getProgram(), "qFirstBackward");
	_qWeightUpdateKernel = cl::Kernel(program.getProgram(), "qWeightUpdate");
	_qLastWeightUpdateKernel = cl::Kernel(program.getProgram(), "qLastWeightUpdate");
	_qActionUpdateKernel = cl::Kernel(program.getProgram(), "qActionUpdate");

	_explorationKernel = cl::Kernel(program.getProgram(), "phExploration");

	// Actions
	_action = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _actionSize.x, _actionSize.y);
	_actionExploratory = createDoubleBuffer2D(cs, _actionSize, CL_R, CL_FLOAT);

	_qFirstErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _actionSize.x, _actionSize.y);

	cs.getQueue().enqueueFillImage(_action, cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 });
	cs.getQueue().enqueueFillImage(_actionExploratory[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 });
}

void AgentHA::simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn) {
	// Feed forward
	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates;

			if (l != 0) {
				visibleStates.resize(2);

				visibleStates[0] = _layers[l - 1]._sc.getHiddenStates()[_back];
				visibleStates[1] = _layers[l]._sc.getHiddenStates()[_front];
			}
			else {
				visibleStates.resize(3);

				visibleStates[0] = input;
				visibleStates[1] = getExploratoryAction();
				visibleStates[2] = _layers[l]._sc.getHiddenStates()[_front];
			}

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);

			// Get reward
			{
				int argIndex = 0;

				_predictionRewardKernel.setArg(argIndex++, _layers[l]._pred.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._reward);

				cs.getQueue().enqueueNDRangeKernel(_predictionRewardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			if (learn)
				_layers[l]._sc.learn(cs, _layers[l]._reward, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
		}
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
	}

	// Find action
	{
		std::vector<cl::Image2D> visibleStates(1);

		visibleStates[0] = _layers.front()._pred.getHiddenStates()[_back];

		_actionPred.activate(cs, visibleStates, false);
	}

	// Copy prediction as starting action
	cs.getQueue().enqueueCopyImage(_actionPred.getHiddenStates()[_back], _action, { 0, 0, 0 }, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 });

#ifdef USE_DETERMINISTIC_POLICY_GRADIENT
	// Find best Q
	float maxQ;

	for (int iter = 0; iter < _actionImprovementIterations; iter++) {
		cl::Image2D prevLayerInput = _action;
		cl_int2 prevLayerSize = _actionSize;

		for (int l = 0; l < _layers.size(); l++) {
			{
				cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_layerDescs[l]._size.x),
					static_cast<float>(prevLayerSize.y) / static_cast<float>(_layerDescs[l]._size.y)
				};

				cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(prevLayerSize.x),
					static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(prevLayerSize.y)
				};

				int argIndex = 0;

				_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
				_qForwardKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
				_qForwardKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
				_qForwardKernel.setArg(argIndex++, prevLayerInput);
				_qForwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
				_qForwardKernel.setArg(argIndex++, prevLayerSize);
				_qForwardKernel.setArg(argIndex++, hiddenToVisible);
				_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
				_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

				cs.getQueue().enqueueNDRangeKernel(_qForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			prevLayerInput = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}

		// Last layer
		{
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_qLastSize.x),
				static_cast<float>(prevLayerSize.y) / static_cast<float>(_qLastSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_qLastSize.x) / static_cast<float>(prevLayerSize.x),
				static_cast<float>(_qLastSize.y) / static_cast<float>(prevLayerSize.y)
			};

			int argIndex = 0;

			_qLastForwardKernel.setArg(argIndex++, _qLastWeights[_back]);
			_qLastForwardKernel.setArg(argIndex++, _qLastBiases[_back]);
			_qLastForwardKernel.setArg(argIndex++, prevLayerInput);
			_qLastForwardKernel.setArg(argIndex++, _qLastStates[_front]);
			_qLastForwardKernel.setArg(argIndex++, prevLayerSize);
			_qLastForwardKernel.setArg(argIndex++, hiddenToVisible);
			_qLastForwardKernel.setArg(argIndex++, _qLastRadius);

			cs.getQueue().enqueueNDRangeKernel(_qLastForwardKernel, cl::NullRange, cl::NDRange(_qLastSize.x, _qLastSize.y));
		}

		if (iter == _actionImprovementIterations - 1) {
			// Find average Q
			float q = 0.0f;

			std::vector<float> qValues(_qLastSize.x * _qLastSize.y);

			cs.getQueue().enqueueReadImage(_qLastStates[_front], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_qLastSize.x), static_cast<cl::size_type>(_qLastSize.y), 1 }, 0, 0, qValues.data());

			for (int i = 0; i < qValues.size(); i++)
				q += qValues[i];

			q /= qValues.size();

			maxQ = q;
		}

		// Backpropagate last layer
		{
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_layerDescs.back()._size.x) / static_cast<float>(_qLastSize.x),
				static_cast<float>(_layerDescs.back()._size.y) / static_cast<float>(_qLastSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_qLastSize.x) / static_cast<float>(_layerDescs.back()._size.x),
				static_cast<float>(_qLastSize.y) / static_cast<float>(_layerDescs.back()._size.y)
			};

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(visibleToHidden.x * (_qLastRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_qLastRadius + 0.5f))) };

			int argIndex = 0;

			_qLastBackwardKernel.setArg(argIndex++, _layers.back()._sc.getHiddenStates()[_back]);
			_qLastBackwardKernel.setArg(argIndex++, _layers.back()._qStates[_front]);
			_qLastBackwardKernel.setArg(argIndex++, _qLastWeights[_back]);
			_qLastBackwardKernel.setArg(argIndex++, _layers.back()._qErrors);
			_qLastBackwardKernel.setArg(argIndex++, _layerDescs.back()._size);
			_qLastBackwardKernel.setArg(argIndex++, _qLastSize);
			_qLastBackwardKernel.setArg(argIndex++, visibleToHidden);
			_qLastBackwardKernel.setArg(argIndex++, hiddenToVisible);
			_qLastBackwardKernel.setArg(argIndex++, _qLastRadius);
			_qLastBackwardKernel.setArg(argIndex++, reverseRadii);
			_qLastBackwardKernel.setArg(argIndex++, _layerDescs.back()._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qLastBackwardKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._size.x, _layerDescs.back()._size.y));
		}

		// Backpropagate other layers
		prevLayerInput = _layers.back()._qErrors;
		prevLayerSize = _layerDescs.back()._size;

		for (int l = _layers.size() - 2; l >= 0; l--) {
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(prevLayerSize.x),
				static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(prevLayerSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_layerDescs[l]._size.x),
				static_cast<float>(prevLayerSize.y) / static_cast<float>(_layerDescs[l]._size.y)
			};

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(visibleToHidden.x * (_layerDescs[l + 1]._qRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_layerDescs[l + 1]._qRadius + 0.5f))) };

			int argIndex = 0;

			_qBackwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_qBackwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
			_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._qWeights[_back]);
			_qBackwardKernel.setArg(argIndex++, prevLayerInput);
			_qBackwardKernel.setArg(argIndex++, _layers[l]._qErrors);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._size);
			_qBackwardKernel.setArg(argIndex++, prevLayerSize);
			_qBackwardKernel.setArg(argIndex++, visibleToHidden);
			_qBackwardKernel.setArg(argIndex++, hiddenToVisible);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l + 1]._qRadius);
			_qBackwardKernel.setArg(argIndex++, reverseRadii);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qBackwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

			prevLayerInput = _layers[l]._qErrors;
			prevLayerSize = _layerDescs[l]._size;
		}

		// Backpropagate to action layer
		{
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_actionSize.x) / static_cast<float>(prevLayerSize.x),
				static_cast<float>(_actionSize.y) / static_cast<float>(prevLayerSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_actionSize.x),
				static_cast<float>(prevLayerSize.y) / static_cast<float>(_actionSize.y)
			};

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(visibleToHidden.x * (_layerDescs.front()._qRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_layerDescs.front()._qRadius + 0.5f))) };

			int argIndex = 0;

			_qFirstBackwardKernel.setArg(argIndex++, _action);
			_qFirstBackwardKernel.setArg(argIndex++, _layers.front()._qWeights[_back]);
			_qFirstBackwardKernel.setArg(argIndex++, prevLayerInput);
			_qFirstBackwardKernel.setArg(argIndex++, _qFirstErrors);
			_qFirstBackwardKernel.setArg(argIndex++, _actionSize);
			_qFirstBackwardKernel.setArg(argIndex++, prevLayerSize);
			_qFirstBackwardKernel.setArg(argIndex++, visibleToHidden);
			_qFirstBackwardKernel.setArg(argIndex++, hiddenToVisible);
			_qFirstBackwardKernel.setArg(argIndex++, _layerDescs.front()._qRadius);
			_qFirstBackwardKernel.setArg(argIndex++, reverseRadii);

			cs.getQueue().enqueueNDRangeKernel(_qFirstBackwardKernel, cl::NullRange, cl::NDRange(_actionSize.x, _actionSize.y));
		}

		// Improve action
		{
			int argIndex = 0;

			_qActionUpdateKernel.setArg(argIndex++, _action);
			_qActionUpdateKernel.setArg(argIndex++, _qFirstErrors);
			_qActionUpdateKernel.setArg(argIndex++, _actionExploratory[_front]);
			_qActionUpdateKernel.setArg(argIndex++, _actionImprovementAlpha);

			cs.getQueue().enqueueNDRangeKernel(_qActionUpdateKernel, cl::NullRange, cl::NDRange(_actionSize.x, _actionSize.y));
		}

		std::swap(_action, _actionExploratory[_front]);
	}

#endif

	// Explore and final activation
	{
		std::uniform_int_distribution<int> seedDist(0, 999);

		cl_uint2 seed = { seedDist(rng), seedDist(rng) };

		int argIndex = 0;

		_explorationKernel.setArg(argIndex++, _action);
		_explorationKernel.setArg(argIndex++, _actionExploratory[_front]);
		_explorationKernel.setArg(argIndex++, _expPert);
		_explorationKernel.setArg(argIndex++, _expBreak);
		_explorationKernel.setArg(argIndex++, seed);

		cs.getQueue().enqueueNDRangeKernel(_explorationKernel, cl::NullRange, cl::NDRange(_actionSize.x, _actionSize.y));
	}

	float tdError;

	// Activate from exploratory actions
	{
		cl::Image2D prevLayerInput = _actionExploratory[_front];
		cl_int2 prevLayerSize = _actionSize;

		for (int l = 0; l < _layers.size(); l++) {
			{
				cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_layerDescs[l]._size.x),
					static_cast<float>(prevLayerSize.y) / static_cast<float>(_layerDescs[l]._size.y)
				};

				cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(prevLayerSize.x),
					static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(prevLayerSize.y)
				};

				int argIndex = 0;

				_qForwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
				_qForwardKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
				_qForwardKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
				_qForwardKernel.setArg(argIndex++, prevLayerInput);
				_qForwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
				_qForwardKernel.setArg(argIndex++, prevLayerSize);
				_qForwardKernel.setArg(argIndex++, hiddenToVisible);
				_qForwardKernel.setArg(argIndex++,_layerDescs[l]._qRadius);
				_qForwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

				cs.getQueue().enqueueNDRangeKernel(_qForwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			prevLayerInput = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}

		// Last layer
		{
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_qLastSize.x),
				static_cast<float>(prevLayerSize.y) / static_cast<float>(_qLastSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_qLastSize.x) / static_cast<float>(prevLayerSize.x),
				static_cast<float>(_qLastSize.y) / static_cast<float>(prevLayerSize.y)
			};

			int argIndex = 0;

			_qLastForwardKernel.setArg(argIndex++, _qLastWeights[_back]);
			_qLastForwardKernel.setArg(argIndex++, _qLastBiases[_back]);
			_qLastForwardKernel.setArg(argIndex++, prevLayerInput);
			_qLastForwardKernel.setArg(argIndex++, _qLastStates[_front]);
			_qLastForwardKernel.setArg(argIndex++, prevLayerSize);
			_qLastForwardKernel.setArg(argIndex++, hiddenToVisible);
			_qLastForwardKernel.setArg(argIndex++, _qLastRadius);

			cs.getQueue().enqueueNDRangeKernel(_qLastForwardKernel, cl::NullRange, cl::NDRange(_qLastSize.x, _qLastSize.y));
		}

		// Find average Q
		float q = 0.0f;

		std::vector<float> qValues(_qLastSize.x * _qLastSize.y);

		cs.getQueue().enqueueReadImage(_qLastStates[_front], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_qLastSize.x), static_cast<cl::size_type>(_qLastSize.y), 1 }, 0, 0, qValues.data());

		for (int i = 0; i < qValues.size(); i++)
			q += qValues[i];

		q /= qValues.size();

		// Bellman equation
		tdError = reward + _qGamma * q - _prevValue;

		std::cout << "Q: " << q << std::endl;

		_prevValue = q;

		// Backpropagate last layer
		{
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_layerDescs.back()._size.x) / static_cast<float>(_qLastSize.x),
				static_cast<float>(_layerDescs.back()._size.y) / static_cast<float>(_qLastSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_qLastSize.x) / static_cast<float>(_layerDescs.back()._size.x),
				static_cast<float>(_qLastSize.y) / static_cast<float>(_layerDescs.back()._size.y)
			};

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(visibleToHidden.x * (_qLastRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_qLastRadius + 0.5f))) };

			int argIndex = 0;

			_qLastBackwardKernel.setArg(argIndex++, _layers.back()._sc.getHiddenStates()[_back]);
			_qLastBackwardKernel.setArg(argIndex++, _layers.back()._qStates[_front]);
			_qLastBackwardKernel.setArg(argIndex++, _qLastWeights[_back]);
			_qLastBackwardKernel.setArg(argIndex++, _layers.back()._qErrors);
			_qLastBackwardKernel.setArg(argIndex++, _layerDescs.back()._size);
			_qLastBackwardKernel.setArg(argIndex++, _qLastSize);
			_qLastBackwardKernel.setArg(argIndex++, visibleToHidden);
			_qLastBackwardKernel.setArg(argIndex++, hiddenToVisible);
			_qLastBackwardKernel.setArg(argIndex++, _qLastRadius);
			_qLastBackwardKernel.setArg(argIndex++, reverseRadii);
			_qLastBackwardKernel.setArg(argIndex++, _layerDescs.back()._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qLastBackwardKernel, cl::NullRange, cl::NDRange(_layerDescs.back()._size.x, _layerDescs.back()._size.y));
		}

		// Backpropagate other layers
		prevLayerInput = _layers.back()._qErrors;
		prevLayerSize = _layerDescs.back()._size;

		for (int l = _layers.size() - 2; l >= 0; l--) {
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(prevLayerSize.x),
				static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(prevLayerSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_layerDescs[l]._size.x),
				static_cast<float>(prevLayerSize.y) / static_cast<float>(_layerDescs[l]._size.y)
			};

			cl_int2 reverseRadii = cl_int2{ static_cast<int>(std::ceil(visibleToHidden.x * (_layerDescs[l + 1]._qRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_layerDescs[l + 1]._qRadius + 0.5f))) };

			int argIndex = 0;

			_qBackwardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
			_qBackwardKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
			_qBackwardKernel.setArg(argIndex++, _layers[l + 1]._qWeights[_back]);
			_qBackwardKernel.setArg(argIndex++, prevLayerInput);
			_qBackwardKernel.setArg(argIndex++, _layers[l]._qErrors);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._size);
			_qBackwardKernel.setArg(argIndex++, prevLayerSize);
			_qBackwardKernel.setArg(argIndex++, visibleToHidden);
			_qBackwardKernel.setArg(argIndex++, hiddenToVisible);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l + 1]._qRadius);
			_qBackwardKernel.setArg(argIndex++, reverseRadii);
			_qBackwardKernel.setArg(argIndex++, _layerDescs[l]._qReluLeak);

			cs.getQueue().enqueueNDRangeKernel(_qBackwardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

			prevLayerInput = _layers[l]._qErrors;
			prevLayerSize = _layerDescs[l]._size;
		}
	}

	// Update weights
	{
		cl::Image2D prevLayerInput = _actionExploratory[_front];
		cl_int2 prevLayerSize = _actionSize;

		for (int l = 0; l < _layers.size(); l++) {
			{
				cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_layerDescs[l]._size.x),
					static_cast<float>(prevLayerSize.y) / static_cast<float>(_layerDescs[l]._size.y)
				};

				cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(prevLayerSize.x),
					static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(prevLayerSize.y)
				};

				int argIndex = 0;

				_qWeightUpdateKernel.setArg(argIndex++, prevLayerInput);
				_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qStates[_front]);
				_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qErrors);
				_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qWeights[_back]);
				_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qWeights[_front]);
				_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qBiases[_back]);
				_qWeightUpdateKernel.setArg(argIndex++, _layers[l]._qBiases[_front]);
				_qWeightUpdateKernel.setArg(argIndex++, prevLayerSize);
				_qWeightUpdateKernel.setArg(argIndex++, hiddenToVisible);
				_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qRadius);
				_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qAlpha);
				_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qBiasAlpha);
				_qWeightUpdateKernel.setArg(argIndex++, _layerDescs[l]._qLambda);
				_qWeightUpdateKernel.setArg(argIndex++, tdError);

				cs.getQueue().enqueueNDRangeKernel(_qWeightUpdateKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			prevLayerInput = _layers[l]._qStates[_front];
			prevLayerSize = _layerDescs[l]._size;
		}

		// Last layer
		{
			cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(prevLayerSize.x) / static_cast<float>(_qLastSize.x),
				static_cast<float>(prevLayerSize.y) / static_cast<float>(_qLastSize.y)
			};

			cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_qLastSize.x) / static_cast<float>(prevLayerSize.x),
				static_cast<float>(_qLastSize.y) / static_cast<float>(prevLayerSize.y)
			};

			int argIndex = 0;

			_qLastWeightUpdateKernel.setArg(argIndex++, prevLayerInput);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastStates[_front]);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastWeights[_back]);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastWeights[_front]);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastBiases[_back]);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastBiases[_front]);
			_qLastWeightUpdateKernel.setArg(argIndex++, prevLayerSize);
			_qLastWeightUpdateKernel.setArg(argIndex++, hiddenToVisible);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastRadius);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastAlpha);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastBiasAlpha);
			_qLastWeightUpdateKernel.setArg(argIndex++, _qLastLambda);
			_qLastWeightUpdateKernel.setArg(argIndex++, tdError);

			cs.getQueue().enqueueNDRangeKernel(_qLastWeightUpdateKernel, cl::NullRange, cl::NDRange(_qLastSize.x, _qLastSize.y));
		}
	}

#ifdef USE_DETERMINISTIC_POLICY_GRADIENT
	if (learn) {
		for (int l = _layers.size() - 1; l >= 0; l--) {
			std::vector<cl::Image2D> visibleStatesPrev;

			if (l < _layers.size() - 1) {
				visibleStatesPrev.resize(2);

				visibleStatesPrev[0] = _layers[l]._sc.getHiddenStates()[_front];
				visibleStatesPrev[1] = _layers[l + 1]._pred.getHiddenStates()[_front];
			}
			else {
				visibleStatesPrev.resize(1);

				visibleStatesPrev[0] = _layers[l]._sc.getHiddenStates()[_front];
			}

			_layers[l]._pred.learn(cs, _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
		}
	
		// Action predictor
		{
			std::vector<cl::Image2D> visibleStates(1);

			visibleStates[0] = _layers.front()._pred.getHiddenStates()[_back];

			_actionPred.learnCurrent(cs, _action, visibleStates, _predActionWeightAlpha);
		}
	}
#else
	if (learn) {
		for (int l = _layers.size() - 1; l >= 0; l--) {
			std::vector<cl::Image2D> visibleStatesPrev;

			if (l < _layers.size() - 1) {
				visibleStatesPrev.resize(2);

				visibleStatesPrev[0] = _layers[l]._sc.getHiddenStates()[_front];
				visibleStatesPrev[1] = _layers[l + 1]._pred.getHiddenStates()[_front];
			}
			else {
				visibleStatesPrev.resize(1);

				visibleStatesPrev[0] = _layers[l]._sc.getHiddenStates()[_front];
			}

			_layers[l]._pred.learn(cs, (tdError > 0.0f ? 1.0f : 0.0f), _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
		}

		// Action predictor
		{
			std::vector<cl::Image2D> visibleStatesPrev(1);

			visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

			_actionPred.learn(cs, (tdError > 0.0f ? 1.0f : 0.0f), _actionExploratory[_back], visibleStatesPrev, _predActionWeightAlpha, _predActionWeightLambda);
		}
	}
#endif

	// Buffer swaps
	{
		for (int l = 0; l < _layers.size(); l++) {
			std::swap(_layers[l]._qStates[_front], _layers[l]._qStates[_back]);
			std::swap(_layers[l]._qWeights[_front], _layers[l]._qWeights[_back]);
			std::swap(_layers[l]._qBiases[_front], _layers[l]._qBiases[_back]);
		}
	
		// Last layer
		std::swap(_qLastStates[_front], _qLastStates[_back]);
		std::swap(_qLastWeights[_front], _qLastWeights[_back]);
		std::swap(_qLastBiases[_front], _qLastBiases[_back]);

		std::swap(_actionExploratory[_front], _actionExploratory[_back]);
	}
}

void AgentHA::clearMemory(sys::ComputeSystem &cs) {
	abort(); // Fix me
	//for (int l = 0; l < _layers.size(); l++)
	//	_layers[l]._sc.clearMemory(cs);
}

void AgentHA::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	abort(); // Not working yet

			 // Layer information
	os << _layers.size() << std::endl;

	for (int li = 0; li < _layers.size(); li++) {
		const Layer &l = _layers[li];
		const LayerDesc &ld = _layerDescs[li];

		// Desc
		os << ld._size.x << " " << ld._size.y << " " << ld._feedForwardRadius << " " << ld._recurrentRadius << " " << ld._lateralRadius << " " << ld._feedBackRadius << " " << ld._predictiveRadius << std::endl;
		//os << ld._scWeightAlpha << " " << ld._scWeightRecurrentAlpha << " " << ld._scWeightLambda << " " << ld._scActiveRatio << " " << ld._scBoostAlpha << std::endl;
		//os << ld._predWeightAlpha << std::endl;

		//l._sc.writeToStream(cs, os);
		//l._pred.writeToStream(cs, os);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			cs.getQueue().enqueueReadImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());

			for (int ri = 0; ri < rewards.size(); ri++)
				os << rewards[ri] << " ";
		}

		os << std::endl;
	}
}

void AgentHA::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
	abort(); // Not working yet

			 // Layer information
	int numLayers;

	is >> numLayers;

	_layers.resize(numLayers);
	_layerDescs.resize(numLayers);

	for (int li = 0; li < _layers.size(); li++) {
		Layer &l = _layers[li];
		LayerDesc &ld = _layerDescs[li];

		// Desc
		is >> ld._size.x >> ld._size.y >> ld._feedForwardRadius >> ld._recurrentRadius >> ld._lateralRadius >> ld._feedBackRadius >> ld._predictiveRadius;
		//is >> ld._scWeightAlpha >> ld._scWeightRecurrentAlpha >> ld._scWeightLambda >> ld._scActiveRatio >> ld._scBoostAlpha;
		//is >> ld._predWeightAlpha;

		l._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		//l._sc.readFromStream(cs, program, is);
		//l._pred.readFromStream(cs, program, is);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			for (int ri = 0; ri < rewards.size(); ri++)
				is >> rewards[ri];

			cs.getQueue().enqueueWriteImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());
		}
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
}
#include "Swarm.h"

using namespace neo;

void Swarm::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 qSize, cl_int2 hiddenSize, int qRadius, cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_visibleLayerDescs = visibleLayerDescs;

	_qSize = qSize;
	_hiddenSize = hiddenSize;
	_qRadius = qRadius;

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> qRegion = { _qSize.x, _qSize.y, 1 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	_visibleLayers.resize(_visibleLayerDescs.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
	cl::Kernel randomUniform3DXYKernel = cl::Kernel(program.getProgram(), "randomUniform3DXY");
	cl::Kernel randomUniform3DXZKernel = cl::Kernel(program.getProgram(), "randomUniform3DXZ");

	// Create layers
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
			static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
		};

		vl._visibleToHidden = cl_float2{ static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
			static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y)
		};

		vl._reverseQRadii = cl_int2{ static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._qRadius)), static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._qRadius)) };

		vl._actions = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);
		vl._actionsExploratory = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);
		vl._predictedAction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		cs.getQueue().enqueueFillImage(vl._actions, zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
		cs.getQueue().enqueueFillImage(vl._actionsExploratory, zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });

		// Q
		{
			int weightDiam = vld._qRadius * 2 + 1;

			int numWeights = weightDiam * weightDiam;

			cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

			vl._qWeights = createDoubleBuffer3D(cs, weightsSize, CL_RGBA, CL_FLOAT);

			randomUniformXZ(vl._qWeights[_back], cs, randomUniform3DXZKernel, weightsSize, initWeightRange, rng);
		}

		// Start
		{
			int weightDiam = vld._startRadius * 2 + 1;

			int numWeights = weightDiam * weightDiam;

			cl_int3 weightsSize = { vld._size.x, vld._size.y, numWeights };

			vl._startWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

			//cs.getQueue().enqueueFillImage(vl._startWeights[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(numWeights) });
			//cs.getQueue().enqueueFillImage(vl._startWeights[_front], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(numWeights) });

			randomUniformXY(vl._startWeights[_back], cs, randomUniform3DXYKernel, weightsSize, initWeightRange, rng);
			randomUniformXY(vl._startWeights[_front], cs, randomUniform3DXYKernel, weightsSize, initWeightRange, rng);
		}
	}

	// Hidden state data
	_qStates = createDoubleBuffer2D(cs, _qSize, CL_R, CL_FLOAT);

	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);
	
	_hiddenErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _hiddenSize.x, _hiddenSize.y);
	_hiddenTD = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _hiddenSize.x, _hiddenSize.y);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, qRegion);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

	{
		int weightDiam = _qRadius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = { _qSize.x, _qSize.y, numWeights };

		_qWeights = createDoubleBuffer3D(cs, weightsSize, CL_RGBA, CL_FLOAT);

		randomUniformXZ(_qWeights[_back], cs, randomUniform3DXZKernel, weightsSize, initWeightRange, rng);
	}

	_qToHidden = cl_float2{ static_cast<float>(_hiddenSize.x) / static_cast<float>(_qSize.x),
		static_cast<float>(_hiddenSize.y) / static_cast<float>(_qSize.y)
	};

	_hiddenToQ = cl_float2{ static_cast<float>(_qSize.x) / static_cast<float>(_hiddenSize.x),
		static_cast<float>(_qSize.y) / static_cast<float>(_hiddenSize.y)
	};

	_reverseQRadii = cl_int2{ static_cast<int>(std::ceil(_hiddenToQ.x * _qRadius)), static_cast<int>(std::ceil(_hiddenToQ.y * _qRadius)) };

	// Create kernels
	_predictAction = cl::Kernel(program.getProgram(), "swarmPredictAction");
	_qActivateToHiddenKernel = cl::Kernel(program.getProgram(), "swarmQActivateToHidden");
	_qActivateToQKernel = cl::Kernel(program.getProgram(), "swarmQActivateToQ");
	_qSolveHiddenKernel = cl::Kernel(program.getProgram(), "swarmQSolveHidden");
	_explorationKernel = cl::Kernel(program.getProgram(), "swarmExploration");
	_qPropagateToHiddenErrorKernel = cl::Kernel(program.getProgram(), "swarmQPropagateToHiddenError");
	_qPropagateToHiddenTDKernel = cl::Kernel(program.getProgram(), "swarmQPropagateToHiddenTD");
	_hiddenPropagateToVisibleActionKernel = cl::Kernel(program.getProgram(), "swarmHiddenPropagateToVisibleAction");
	_startLearnWeightsKernel = cl::Kernel(program.getProgram(), "swarmStartLearnWeights");
	_qLearnVisibleWeightsTracesKernel = cl::Kernel(program.getProgram(), "swarmQLearnVisibleWeightsTraces");
	_qLearnHiddenWeightsTracesKernel = cl::Kernel(program.getProgram(), "swarmQLearnHiddenWeightsTraces");
}

void Swarm::simStep(sys::ComputeSystem &cs, float reward,
	const cl::Image2D &hiddenStatesFeedForward, const cl::Image2D &actionsFeedBack,
	float expPert, float expBreak, int annealIterations, float actionAlpha,
	float alphaHiddenQ, float alphaQ, float alphaPred, float lambda, float gamma, std::mt19937 &rng)
{
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	// Find Q errors (same for everything this tick, so calculate once)
	{
		int argIndex = 0;

		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _qWeights[_back]);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _hiddenErrors);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _qSize);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _hiddenSize);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _qToHidden);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _hiddenToQ);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _qRadius);
		_qPropagateToHiddenErrorKernel.setArg(argIndex++, _reverseQRadii);

		cs.getQueue().enqueueNDRangeKernel(_qPropagateToHiddenErrorKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}
	
	// Find starting action by activating action predictors from hidden state
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		cl::array<cl::size_type, 3> visibleRegion = { vld._size.x, vld._size.y, 1 };

		int argIndex = 0;

		_predictAction.setArg(argIndex++, hiddenStatesFeedForward);
		_predictAction.setArg(argIndex++, actionsFeedBack);
		_predictAction.setArg(argIndex++, vl._startWeights[_back]);
		_predictAction.setArg(argIndex++, vl._predictedAction);
		_predictAction.setArg(argIndex++, _hiddenSize);
		_predictAction.setArg(argIndex++, vl._visibleToHidden);
		_predictAction.setArg(argIndex++, vld._startRadius);

		cs.getQueue().enqueueNDRangeKernel(_predictAction, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));

		// Copy as a starting point
		cs.getQueue().enqueueCopyImage(vl._predictedAction, vl._actions, zeroOrigin, zeroOrigin, visibleRegion);
	}

	// Anneal actions
	for (int iter = 0; iter < annealIterations; iter++) {
		// Start by clearing summation buffer
		cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);

		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
			VisibleLayer &vl = _visibleLayers[vli];
			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

			int argIndex = 0;

			_qActivateToHiddenKernel.setArg(argIndex++, vl._actions);
			_qActivateToHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_qActivateToHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
			_qActivateToHiddenKernel.setArg(argIndex++, vl._qWeights[_back]);
			_qActivateToHiddenKernel.setArg(argIndex++, vld._size);
			_qActivateToHiddenKernel.setArg(argIndex++, vl._hiddenToVisible);
			_qActivateToHiddenKernel.setArg(argIndex++, vld._qRadius);

			cs.getQueue().enqueueNDRangeKernel(_qActivateToHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			// Swap buffers
			std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
		}

		{
			std::uniform_int_distribution<int> seedDist;

			cl_uint2 seed = { seedDist(rng), seedDist(rng) };

			int argIndex = 0;

			_qSolveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_qSolveHiddenKernel.setArg(argIndex++, hiddenStatesFeedForward);
			_qSolveHiddenKernel.setArg(argIndex++, actionsFeedBack);
			_qSolveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);

			cs.getQueue().enqueueNDRangeKernel(_qSolveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		// Backpropagate
		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
			VisibleLayer &vl = _visibleLayers[vli];
			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

			int argIndex = 0;

			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, _hiddenErrors);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, _hiddenStates[_front]);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vl._qWeights[_back]);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vl._actions);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vl._actionsExploratory); // Use exploratory actions buffer temporarily here, not used yet anyways
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, _hiddenSize);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vld._size);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vl._hiddenToVisible);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vl._visibleToHidden);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vld._qRadius);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, vl._reverseQRadii);
			_hiddenPropagateToVisibleActionKernel.setArg(argIndex++, actionAlpha);

			cs.getQueue().enqueueNDRangeKernel(_hiddenPropagateToVisibleActionKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
		
			std::swap(vl._actions, vl._actionsExploratory);
		}
	}

	// Exploration
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		std::uniform_int_distribution<int> seedDist;

		cl_uint2 seed = { seedDist(rng), seedDist(rng) };

		int argIndex = 0;

		_explorationKernel.setArg(argIndex++, vl._actions);
		_explorationKernel.setArg(argIndex++, vl._actionsExploratory);
		_explorationKernel.setArg(argIndex++, expPert);
		_explorationKernel.setArg(argIndex++, expBreak);
		_explorationKernel.setArg(argIndex++, seed);

		cs.getQueue().enqueueNDRangeKernel(_explorationKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}

	// Activate from exploratory action
	{
		// Start by clearing summation buffer
		cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);

		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
			VisibleLayer &vl = _visibleLayers[vli];
			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

			int argIndex = 0;

			_qActivateToHiddenKernel.setArg(argIndex++, vl._actionsExploratory); // Use exploratory action now
			_qActivateToHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_qActivateToHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
			_qActivateToHiddenKernel.setArg(argIndex++, vl._qWeights[_back]);
			_qActivateToHiddenKernel.setArg(argIndex++, vld._size);
			_qActivateToHiddenKernel.setArg(argIndex++, vl._hiddenToVisible);
			_qActivateToHiddenKernel.setArg(argIndex++, vld._qRadius);

			cs.getQueue().enqueueNDRangeKernel(_qActivateToHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			// Swap buffers
			std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
		}

		{
			std::uniform_int_distribution<int> seedDist;

			cl_uint2 seed = { seedDist(rng), seedDist(rng) };

			int argIndex = 0;

			_qSolveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_qSolveHiddenKernel.setArg(argIndex++, hiddenStatesFeedForward);
			_qSolveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
	
			cs.getQueue().enqueueNDRangeKernel(_qSolveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}
	}

	// Find Q
	{
		int argIndex = 0;

		_qActivateToQKernel.setArg(argIndex++, _hiddenStates[_front]);
		_qActivateToQKernel.setArg(argIndex++, _qWeights[_back]);
		_qActivateToQKernel.setArg(argIndex++, _qStates[_front]);
		_qActivateToQKernel.setArg(argIndex++, _hiddenSize);
		_qActivateToQKernel.setArg(argIndex++, _qToHidden);
		_qActivateToQKernel.setArg(argIndex++, _qRadius);

		cs.getQueue().enqueueNDRangeKernel(_qActivateToQKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Find TD errors
	{
		int argIndex = 0;

		_qPropagateToHiddenTDKernel.setArg(argIndex++, _qStates[_front]);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _qStates[_back]);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _hiddenTD);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _qSize);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _hiddenSize);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _qToHidden);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _hiddenToQ);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _qRadius);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, _reverseQRadii);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, reward);
		_qPropagateToHiddenTDKernel.setArg(argIndex++, gamma);

		cs.getQueue().enqueueNDRangeKernel(_qPropagateToHiddenTDKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Weight updates
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		{
			int argIndex = 0;

			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, vl._actionsExploratory);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, _hiddenErrors);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, _hiddenTD);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_front]);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, vl._qWeights[_back]);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, vl._qWeights[_front]);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, vld._size);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, vld._qRadius);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, alphaHiddenQ);
			_qLearnVisibleWeightsTracesKernel.setArg(argIndex++, lambda);

			cs.getQueue().enqueueNDRangeKernel(_qLearnVisibleWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		{
			int argIndex = 0;

			_startLearnWeightsKernel.setArg(argIndex++, vl._actions);
			_startLearnWeightsKernel.setArg(argIndex++, vl._predictedAction);
			_startLearnWeightsKernel.setArg(argIndex++, hiddenStatesFeedForward);
			_startLearnWeightsKernel.setArg(argIndex++, actionsFeedBack);
			_startLearnWeightsKernel.setArg(argIndex++, vl._startWeights[_back]);
			_startLearnWeightsKernel.setArg(argIndex++, vl._startWeights[_front]);
			_startLearnWeightsKernel.setArg(argIndex++, _hiddenSize);
			_startLearnWeightsKernel.setArg(argIndex++, vl._visibleToHidden);
			_startLearnWeightsKernel.setArg(argIndex++, vld._startRadius);
			_startLearnWeightsKernel.setArg(argIndex++, alphaPred);

			cs.getQueue().enqueueNDRangeKernel(_startLearnWeightsKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
		}
	}

	// Learn Q weights
	{
		int argIndex = 0;

		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_front]);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _qStates[_front]);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _qStates[_back]);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _qWeights[_back]);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _qWeights[_front]);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _hiddenSize);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _qToHidden);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, _qRadius);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, alphaQ);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, lambda);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, reward);
		_qLearnHiddenWeightsTracesKernel.setArg(argIndex++, gamma);

		cs.getQueue().enqueueNDRangeKernel(_qLearnHiddenWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Swap buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);

	std::swap(_qStates[_front], _qStates[_back]);
	
	std::swap(_qWeights[_front], _qWeights[_back]);

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		std::swap(vl._qWeights[_front], vl._qWeights[_back]);
		std::swap(vl._startWeights[_front], vl._startWeights[_back]);
	}
}
#include "AgentER.h"

#include <iostream>

using namespace neo;

void AgentER::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int2 qSize,
	const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_inputSize = inputSize;
	_actionSize = actionSize;
	_qSize = qSize;

	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	cl::Kernel randomUniform2DXYKernel = cl::Kernel(program.getProgram(), "randomUniform2DXY");

	cl_int2 prevLayerSize = inputSize;

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs;

		if (l == 0) {
			scDescs.resize(3);

			scDescs[0]._size = prevLayerSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._useTraces = false;

			scDescs[1]._size = _actionSize;
			scDescs[1]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[1]._ignoreMiddle = false;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[1]._useTraces = false;

			scDescs[2]._size = _qSize;
			scDescs[2]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[2]._ignoreMiddle = false;
			scDescs[2]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[2]._useTraces = false;
		}
		else {
			scDescs.resize(2);

			scDescs[0]._size = prevLayerSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._useTraces = false;

			scDescs[1]._size = _layerDescs[l]._size;
			scDescs[1]._radius = _layerDescs[l]._recurrentRadius;
			scDescs[1]._ignoreMiddle = true;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
			scDescs[1]._useTraces = false;
		}

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, rng);

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

		if (l == 0)
			_layers[l]._pred.createRandom(cs, program, predDescs, _actionSize, initWeightRange, true, rng);
		else
			_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l - 1]._size, initWeightRange, true, rng);

		// Create baselines
		_layers[l]._predReward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);
		_layers[l]._propagatedPredReward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._predReward, zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._propagatedPredReward, zeroColor, zeroOrigin, layerRegion);

		_layers[l]._scStatesTemp = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);
		_layers[l]._predStatesTemp = createDoubleBuffer2D(cs, prevLayerSize, CL_R, CL_FLOAT);

		prevLayerSize = _layerDescs[l]._size;
	}

	_qInput = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _qSize.x, _qSize.y);

	_qTarget = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _qSize.x, _qSize.y);

	_actionTarget = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _actionSize.x, _actionSize.y);

	_qTransform = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _qSize.x, _qSize.y);

	// Q Predictor
	{
		std::vector<Predictor::VisibleLayerDesc> predDescs;

		if (0 < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[0]._size;
			predDescs[0]._radius = _layerDescs[0]._predictiveRadius;

			predDescs[1]._size = _layerDescs[0 + 1]._size;
			predDescs[1]._radius = _layerDescs[0]._feedBackRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[0]._size;
			predDescs[0]._radius = _layerDescs[0]._predictiveRadius;
		}

		_qPred.createRandom(cs, program, predDescs, _qSize, initWeightRange, true, rng);
	}

	// Random Q transform
	randomUniformXY(_qTransform, cs, randomUniform2DXYKernel, _qSize, { -1.0f, 1.0f }, rng);

	_inputWhitener.create(cs, program, _inputSize, CL_R, CL_FLOAT);
	_actionWhitener.create(cs, program, _actionSize, CL_R, CL_FLOAT);
	_qWhitener.create(cs, program, _qSize, CL_R, CL_FLOAT);

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
	_predictionRewardPropagationKernel = cl::Kernel(program.getProgram(), "phPredictionRewardPropagation");
	_setQKernel = cl::Kernel(program.getProgram(), "phSetQ");
}

void AgentER::simStep(sys::ComputeSystem &cs, const cl::Image2D &input, const cl::Image2D &actionTaken, float reward, std::mt19937 &rng, bool learn, bool whiten) {
	// Keep previous best action for later
	std::vector<float> prevBestAction(_actionSize.x * _actionSize.y);
	std::vector<float> prevTakenAction(_actionSize.x * _actionSize.y);

	cs.getQueue().enqueueReadImage(getAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 }, 0, 0, prevBestAction.data());
	cs.getQueue().enqueueReadImage(actionTaken, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 }, 0, 0, prevTakenAction.data());

	// Place previous Q into Q buffer
	{
		int argIndex = 0;

		_setQKernel.setArg(argIndex++, _qTransform);
		_setQKernel.setArg(argIndex++, _qInput);
		_setQKernel.setArg(argIndex++, _prevQ);

		cs.getQueue().enqueueNDRangeKernel(_setQKernel, cl::NullRange, cl::NDRange(_qSize.x, _qSize.y));
	}

	// Whiten input
	if (whiten)
		_inputWhitener.filter(cs, input, _whiteningKernelRadius, _whiteningIntensity);

	_actionWhitener.filter(cs, actionTaken, _whiteningKernelRadius, _whiteningIntensity);

	_qWhitener.filter(cs, _qInput, _whiteningKernelRadius, _whiteningIntensity);

	// Feed forward
	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates;

			if (l == 0) {
				visibleStates.resize(3);

				visibleStates[0] = whiten ? _inputWhitener.getResult() : input;
				visibleStates[1] = _actionWhitener.getResult();
				visibleStates[2] = _qWhitener.getResult();
			}
			else {
				visibleStates.resize(2);

				visibleStates[0] = _layers[l - 1]._sc.getHiddenStates()[_back];
				visibleStates[1] = _layers[l]._sc.getHiddenStates()[_back];
			}

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);
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

		_layers[l]._pred.activate(cs, visibleStates, l != 0);
	}

	// Q predictor
	{
		std::vector<cl::Image2D> visibleStates;

		if (0 < _layers.size() - 1) {
			visibleStates.resize(2);

			visibleStates[0] = _layers[0]._sc.getHiddenStates()[_back];
			visibleStates[1] = _layers[0 + 1]._pred.getHiddenStates()[_back];
		}
		else {
			visibleStates.resize(1);

			visibleStates[0] = _layers[0]._sc.getHiddenStates()[_back];
		}

		_qPred.activate(cs, visibleStates, false);
	}

	// Recover Q
	std::vector<float> qValues(_qSize.x * _qSize.y);

	cs.getQueue().enqueueReadImage(_qPred.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_qSize.x), static_cast<cl::size_type>(_qSize.y), 1 }, 0, 0, qValues.data());

	// Average all Q values
	float q = 0.0f;

	for (int i = 0; i < qValues.size(); i++)
		q += qValues[i];

	q /= qValues.size();

	// Bellman equation
	float tdError = reward + _qGamma * q - _prevValue;

	float newQ = _prevValue + _qAlpha * tdError;

	// Update older samples
	float g = _qGamma;

	for (std::list<ReplayFrame>::iterator it = _frames.begin(); it != _frames.end(); it++) {
		it->_q += g * tdError;

		g *= _qGamma;
	}

	// Add replay sample
	ReplayFrame frame;

	frame._q = frame._originalQ = newQ;

	frame._layerStateBitIndices.resize(_layers.size());
	frame._layerPredBitIndices.resize(_layers.size());

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<float> state(_layerDescs[l]._size.x * _layerDescs[l]._size.y);

		cs.getQueue().enqueueReadImage(_layers[l]._sc.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l]._size.x), static_cast<cl::size_type>(_layerDescs[l]._size.y), 1 }, 0, 0, state.data());
	
		std::vector<float> pred;
		
		if (l == 0) {
			pred.resize(_actionSize.x * _actionSize.y);

			cs.getQueue().enqueueReadImage(_layers[l]._sc.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 }, 0, 0, state.data());
		}
		else {
			pred.resize(_layerDescs[l - 1]._size.x * _layerDescs[l - 1]._size.y);

			cs.getQueue().enqueueReadImage(_layers[l]._sc.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l - 1]._size.x), static_cast<cl::size_type>(_layerDescs[l - 1]._size.y), 1 }, 0, 0, pred.data());
		}

		for (int i = 0; i < state.size(); i++)
			if (state[i] > 0.0f)
				frame._layerStateBitIndices[l].push_back(i);

		for (int i = 0; i < pred.size(); i++)
			if (pred[i] > 0.0f)
				frame._layerPredBitIndices[l].push_back(i);
	}

	// Add last action taken and last "thought best" action
	frame._prevExploratoryAction = prevTakenAction;
	frame._prevBestAction = prevBestAction;

	for (int i = 0; i < prevBestAction.size(); i++)
		frame._prevBestAction[i] = std::min(1.0f, std::max(-1.0f, prevBestAction[i]));

	_frames.push_front(frame);

	while (_frames.size() > _maxReplayFrames)
		_frames.pop_back();

	if (learn && _frames.size() > 1) {
		// Convert list to vector
		std::vector<ReplayFrame*> pFrames(_frames.size());

		int index = 0;

		for (std::list<ReplayFrame>::iterator it = _frames.begin(); it != _frames.end(); it++)
			pFrames[index++] = &(*it);

		std::uniform_int_distribution<int> replayDist(0, _frames.size() - 2);

		for (int iter = 0; iter < _replayIterations; iter++) {
			int randIndex = replayDist(rng);

			ReplayFrame* pFrame = pFrames[randIndex];
			ReplayFrame* pFramePrev = pFrames[randIndex + 1];

			// Load data
			cl_int2 prevLayerSize = _actionSize;

			for (int l = 0; l < _layers.size(); l++) {
				std::vector<float> state(_layerDescs[l]._size.x * _layerDescs[l]._size.y, 0.0f);
				std::vector<float> statePrev(_layerDescs[l]._size.x * _layerDescs[l]._size.y, 0.0f);
				std::vector<float> pred(prevLayerSize.x * prevLayerSize.y, 0.0f);
				std::vector<float> predPrev(prevLayerSize.x * prevLayerSize.y, 0.0f);

				for (int i = 0; i < pFrame->_layerStateBitIndices[l].size(); i++)
					state[pFrame->_layerStateBitIndices[l][i]] = 1.0f;

				for (int i = 0; i < pFramePrev->_layerStateBitIndices[l].size(); i++)
					statePrev[pFramePrev->_layerStateBitIndices[l][i]] = 1.0f;

				for (int i = 0; i < pFrame->_layerPredBitIndices[l].size(); i++)
					pred[pFrame->_layerPredBitIndices[l][i]] = 1.0f;

				for (int i = 0; i < pFramePrev->_layerPredBitIndices[l].size(); i++)
					predPrev[pFramePrev->_layerPredBitIndices[l][i]] = 1.0f;

				cs.getQueue().enqueueWriteImage(_layers[l]._scStatesTemp[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l]._size.x), static_cast<cl::size_type>(_layerDescs[l]._size.y), 1 }, 0, 0, state.data());
				cs.getQueue().enqueueWriteImage(_layers[l]._scStatesTemp[_front], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l]._size.x), static_cast<cl::size_type>(_layerDescs[l]._size.y), 1 }, 0, 0, statePrev.data());
			
				cs.getQueue().enqueueWriteImage(_layers[l]._predStatesTemp[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(prevLayerSize.x), static_cast<cl::size_type>(prevLayerSize.y), 1 }, 0, 0, pred.data());
				cs.getQueue().enqueueWriteImage(_layers[l]._predStatesTemp[_front], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(prevLayerSize.x), static_cast<cl::size_type>(prevLayerSize.y), 1 }, 0, 0, predPrev.data());

				prevLayerSize = _layerDescs[l]._size;
			}

			cs.getQueue().enqueueFillImage(_qTarget, cl_float4{ pFrame->_q, pFrame->_q, pFrame->_q, pFrame->_q }, { 0, 0, 0 }, { static_cast<cl::size_type>(_qSize.x), static_cast<cl::size_type>(_qSize.y), 1 });
			
			// Choose better action to learn
			cs.getQueue().enqueueWriteImage(_actionTarget, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionSize.x), static_cast<cl::size_type>(_actionSize.y), 1 }, 0, 0,
				(pFrame->_q > pFrame->_originalQ ? pFrame->_prevExploratoryAction.data() : pFrame->_prevBestAction.data()));

			for (int l = 0; l < _layers.size(); l++) {
				std::vector<cl::Image2D> visibleStates;

				if (l != 0) {
					visibleStates.resize(2);

					visibleStates[0] = _layers[l - 1]._sc.getHiddenStates()[_back];
					visibleStates[1] = _layers[l]._sc.getHiddenStates()[_back];

					_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio, false);

					_layers[l]._sc.learn(cs, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
				}

				std::vector<cl::Image2D> visibleStatesPrev;

				if (l < _layers.size() - 1) {
					visibleStatesPrev.resize(2);

					visibleStatesPrev[0] = _layers[l]._scStatesTemp[_front];
					visibleStatesPrev[1] = _layers[l + 1]._predStatesTemp[_front];
				}
				else {
					visibleStatesPrev.resize(1);

					visibleStatesPrev[0] = _layers[l]._scStatesTemp[_front];
				}

				_layers[l]._pred.activate(cs, visibleStatesPrev, l != 0, false);

				if (l == 0)
					_layers[l]._pred.learnCurrent(cs, _actionTarget, visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
				else
					_layers[l]._pred.learnCurrent(cs, _layers[l - 1]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
			}

			// Q Pred
			{
				std::vector<cl::Image2D> visibleStatesPrev;

				if (0 < _layers.size() - 1) {
					visibleStatesPrev.resize(2);

					visibleStatesPrev[0] = _layers[0]._sc.getHiddenStates()[_front];
					visibleStatesPrev[1] = _layers[0 + 1]._pred.getHiddenStates()[_front];
				}
				else {
					visibleStatesPrev.resize(1);

					visibleStatesPrev[0] = _layers[0]._sc.getHiddenStates()[_front];
				}

				_qPred.activate(cs, visibleStatesPrev, false, false);

				_qPred.learnCurrent(cs, _qTarget, visibleStatesPrev, _qWeightAlpha);
			}
		}
	}

	std::cout << "Q: " << newQ << std::endl;

	_prevQ = newQ;
	_prevTDError = tdError;
	_prevValue = q;
}

void AgentER::clearMemory(sys::ComputeSystem &cs) {
	// Fix me
	abort();
}

void AgentER::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	abort(); // Not working yet

			 // Layer information
	os << _layers.size() << std::endl;

	for (int li = 0; li < _layers.size(); li++) {
		const Layer &l = _layers[li];
		const LayerDesc &ld = _layerDescs[li];

		// Desc
		os << ld._size.x << " " << ld._size.y << " " << ld._feedForwardRadius << " " << ld._recurrentRadius << " " << ld._lateralRadius << " " << ld._feedBackRadius << " " << ld._predictiveRadius << std::endl;
		//os << ld._scWeightAlpha << " " << ld._scWeightRecurrentAlpha << " " << ld._scWeightLambda << " " << ld._scActiveRatio << " " << ld._scBoostAlpha << std::endl;
		os << ld._predWeightAlpha << std::endl;

		l._sc.writeToStream(cs, os);
		l._pred.writeToStream(cs, os);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			//cs.getQueue().enqueueReadImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());

			for (int ri = 0; ri < rewards.size(); ri++)
				os << rewards[ri] << " ";
		}

		os << std::endl;

		{
			std::vector<cl_float> hiddenStatesPrev(ld._size.x * ld._size.y);

			//cs.getQueue().enqueueReadImage(l._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, hiddenStatesPrev.data());

			for (int si = 0; si < hiddenStatesPrev.size(); si++)
				os << hiddenStatesPrev[si] << " ";
		}

		os << std::endl;
	}
}

void AgentER::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
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
		is >> ld._predWeightAlpha;

		//l._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		//l._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		l._sc.readFromStream(cs, program, is);
		l._pred.readFromStream(cs, program, is);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			for (int ri = 0; ri < rewards.size(); ri++)
				is >> rewards[ri];

			//cs.getQueue().enqueueWriteImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());
		}

		{
			std::vector<cl_float> hiddenStatesPrev(ld._size.x * ld._size.y);

			for (int si = 0; si < hiddenStatesPrev.size(); si++)
				is >> hiddenStatesPrev[si];

			//cs.getQueue().enqueueWriteImage(l._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, hiddenStatesPrev.data());
		}
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
}
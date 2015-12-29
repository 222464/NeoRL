#include "AgentSPG.h"

using namespace neo;

void AgentSPG::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerFeedBackRadius, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_inputSize = inputSize;
	_actionSize = actionSize;

	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

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

		std::vector<PredictorSwarm::VisibleLayerDesc> predDescs;

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

		_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l]._size, initWeightRange, rng);

		// Create baselines
		_layers[l]._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		_layers[l]._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._reward, zeroColor, zeroOrigin, layerRegion);
	}

	// First layer stuff
	{
		std::vector<PredictorSwarm::VisibleLayerDesc> predDescs;

		assert(!_layers.empty());

		predDescs.resize(1);

		predDescs[0]._size = _layerDescs.front()._size;
		predDescs[0]._radius = firstLayerFeedBackRadius;

		_firstLayerPred.createRandom(cs, program, predDescs, actionSize, initWeightRange, rng);
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
}

void AgentSPG::simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn) {
	// Feed forward
	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates;

			if (l != 0) {
				visibleStates.resize(2);

				visibleStates[0] = _layers[l - 1]._sc.getHiddenStates()[_back];
				visibleStates[1] = _layers[l]._scHiddenStatesPrev;
			}
			else {
				visibleStates.resize(3);

				visibleStates[0] = input;
				visibleStates[1] = _firstLayerPred.getHiddenStates()[_back];
				visibleStates[2] = _layers[l]._scHiddenStatesPrev;
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

		_layers[l]._pred.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio, _layerDescs[l]._lateralRadius, _layerDescs[l]._noise, rng);
	}

	// First layer
	{
		std::vector<cl::Image2D> visibleStates;

		visibleStates.resize(1);

		visibleStates[0] = _layers.front()._pred.getHiddenStates()[_back];

		_firstLayerPred.activate(cs, visibleStates, 1.0f, -1, _firstLayerNoise, rng);
	}

	if (learn) {
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

			_layers[l]._pred.learn(cs, reward, _layerDescs[l]._gamma, _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._lambda);
		}

		// First layer
		{
			std::vector<cl::Image2D> visibleStatesPrev;

			visibleStatesPrev.resize(1);

			visibleStatesPrev[0] = _layers.front()._pred.getHiddenStates()[_front];

			_firstLayerPred.learn(cs, reward, _firstLayerGamma, _firstLayerPred.getHiddenStates()[_back], visibleStatesPrev, _firstLayerPredWeightAlpha, _firstLayerLambda);
		}
	}

	// Buffer updates
	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueCopyImage(_layers[l]._sc.getHiddenStates()[_back], _layers[l]._scHiddenStatesPrev, zeroOrigin, zeroOrigin, layerRegion);
	}
}

void AgentSPG::clearMemory(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

	for (int l = 0; l < _layers.size(); l++) {
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._scHiddenStatesPrev, zeroColor, zeroOrigin, layerRegion);
	}
}

void AgentSPG::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	abort(); // Not working yet

	// Layer information
	os << _layers.size() << std::endl;

	for (int li = 0; li < _layers.size(); li++) {
		const Layer &l = _layers[li];
		const LayerDesc &ld = _layerDescs[li];

		// Desc
		os << ld._size.x << " " << ld._size.y << " " << ld._feedForwardRadius << " " << ld._recurrentRadius << " " << ld._lateralRadius << " " << ld._feedBackRadius << " " << ld._predictiveRadius << std::endl;
		os << ld._scWeightAlpha << " " << ld._scWeightRecurrentAlpha << " " << ld._scWeightLambda << " " << ld._scActiveRatio << " " << ld._scBoostAlpha << std::endl;
		//os << ld._predWeightAlpha << std::endl;

		l._sc.writeToStream(cs, os);
		//l._pred.writeToStream(cs, os);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			cs.getQueue().enqueueReadImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());

			for (int ri = 0; ri < rewards.size(); ri++)
				os << rewards[ri] << " ";
		}

		os << std::endl;

		{
			std::vector<cl_float> hiddenStatesPrev(ld._size.x * ld._size.y);

			cs.getQueue().enqueueReadImage(l._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, hiddenStatesPrev.data());

			for (int si = 0; si < hiddenStatesPrev.size(); si++)
				os << hiddenStatesPrev[si] << " ";
		}

		os << std::endl;
	}
}

void AgentSPG::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
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
		is >> ld._scWeightAlpha >> ld._scWeightRecurrentAlpha >> ld._scWeightLambda >> ld._scActiveRatio >> ld._scBoostAlpha;
		//is >> ld._predWeightAlpha;

		l._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		l._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		l._sc.readFromStream(cs, program, is);
		//l._pred.readFromStream(cs, program, is);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			for (int ri = 0; ri < rewards.size(); ri++)
				is >> rewards[ri];

			cs.getQueue().enqueueWriteImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());
		}

		{
			std::vector<cl_float> hiddenStatesPrev(ld._size.x * ld._size.y);

			for (int si = 0; si < hiddenStatesPrev.size(); si++)
				is >> hiddenStatesPrev[si];

			cs.getQueue().enqueueWriteImage(l._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, hiddenStatesPrev.data());
		}
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
}
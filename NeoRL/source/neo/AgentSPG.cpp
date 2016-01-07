#include "AgentSPG.h"

using namespace neo;

void AgentSPG::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_inputSize = inputSize;
	_actionSize = actionSize;

	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");

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
			scDescs.resize(2);

			scDescs[0]._size = _inputSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[0]._useTraces = false;

			scDescs[1]._size = _actionSize;
			scDescs[1]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[1]._ignoreMiddle = false;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[1]._useTraces = false;

			/*scDescs[2]._size = _layerDescs[l]._size;
			scDescs[2]._radius = _layerDescs[l]._recurrentRadius;
			scDescs[2]._ignoreMiddle = true;
			scDescs[2]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
			scDescs[2]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[2]._useTraces = true;*/
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
		_layers[l]._predReward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);
		_layers[l]._propagatedPredReward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._predReward, zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._propagatedPredReward, zeroColor, zeroOrigin, layerRegion);
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
	_predictionRewardPropagationKernel = cl::Kernel(program.getProgram(), "phPredictionRewardPropagation");

	_inputWhitener.create(cs, program, _inputSize, CL_R, CL_FLOAT);
	_actionWhitener.create(cs, program, _actionSize, CL_R, CL_FLOAT);
}

void AgentSPG::simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn) {
	// Whiten input
	_inputWhitener.filter(cs, input, _whiteningKernelRadius, _whiteningIntensity);
	_actionWhitener.filter(cs, getExploratoryAction(), _whiteningKernelRadius, _whiteningIntensity);

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
				visibleStates.resize(2);

				visibleStates[0] = _inputWhitener.getResult();
				visibleStates[1] = _actionWhitener.getResult();
				//visibleStates[2] = _layers[l]._sc.getHiddenStates()[_front];
			}

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);

			// Get reward
			if (l < _layers.size() - 1) {
				int argIndex = 0;

				_predictionRewardKernel.setArg(argIndex++, _layers[l + 1]._pred.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predReward);
				_predictionRewardKernel.setArg(argIndex++, _layerDescs[l]._scActiveRatio);

				cs.getQueue().enqueueNDRangeKernel(_predictionRewardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			// Propagate reward
			if (l != 0) {
				// Propagate to first target
				cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_layerDescs[l - 1]._size.x) / static_cast<float>(_layerDescs[l]._size.x),
					static_cast<float>(_layerDescs[l - 1]._size.y) / static_cast<float>(_layerDescs[l]._size.y)
				};

				cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(_layerDescs[l - 1]._size.x),
					static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(_layerDescs[l - 1]._size.y)
				};

				cl_int radius = std::max(static_cast<int>(std::ceil(visibleToHidden.x * (_layerDescs[l]._predictiveRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_layerDescs[l]._predictiveRadius + 0.5f))));


				int argIndex = 0;

				_predictionRewardPropagationKernel.setArg(argIndex++, _layers[l - 1]._predReward);
				_predictionRewardPropagationKernel.setArg(argIndex++, _layers[l]._propagatedPredReward);
				_predictionRewardPropagationKernel.setArg(argIndex++, hiddenToVisible);
				_predictionRewardPropagationKernel.setArg(argIndex++, _layerDescs[l - 1]._size);
				_predictionRewardPropagationKernel.setArg(argIndex++, radius);

				cs.getQueue().enqueueNDRangeKernel(_predictionRewardPropagationKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			if (learn) {
				if (l == 0)
					_layers[l]._sc.learn(cs, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
				else
					_layers[l]._sc.learn(cs, _layers[l]._propagatedPredReward, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
			}
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

		if (l == 0)
			_layers[l]._pred.activate(cs, visibleStates, _layerDescs[l]._noise, false, rng);
		else
			_layers[l]._pred.activateInhibit(cs, visibleStates, _layerDescs[l]._noise, _layerDescs[l]._scActiveRatio, _layerDescs[l]._lateralRadius, rng);
	}

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

			if (l == 0)
				_layers[l]._pred.learn(cs, reward, _layerDescs[l]._gamma, _layers.front()._pred.getHiddenStates()[_front], visibleStatesPrev, _layerDescs[l]._alpha, _layerDescs[l]._lambda);
			else
				_layers[l]._pred.learn(cs, reward, _layerDescs[l]._gamma, _layers[l]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._alpha, _layerDescs[l]._lambda);
		}
	}
}

void AgentSPG::clearMemory(sys::ComputeSystem &cs) {
	for (int l = 0; l < _layers.size(); l++)
		_layers[l]._sc.clearMemory(cs);
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

			//cs.getQueue().enqueueReadImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());

			for (int ri = 0; ri < rewards.size(); ri++)
				os << rewards[ri] << " ";
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

		//l._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		l._sc.readFromStream(cs, program, is);
		//l._pred.readFromStream(cs, program, is);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			for (int ri = 0; ri < rewards.size(); ri++)
				is >> rewards[ri];

			//cs.getQueue().enqueueWriteImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());
		}
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
}
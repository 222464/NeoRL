#include "PredictiveHierarchy.h"

using namespace neo;

void PredictiveHierarchy::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_inputSize = inputSize;

	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	cl_int2 prevLayerSize = inputSize;

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<SparsePredictor::VisibleLayerDesc> scDescs;

		if (l == 0) {
			scDescs.resize(2);

			scDescs[0]._size = prevLayerSize;
			scDescs[0]._encodeRadius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			scDescs[0]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			scDescs[0]._predictBinary = false;

			scDescs[1]._size = _layerDescs[l]._size;
			scDescs[1]._encodeRadius = _layerDescs[l]._recurrentRadius;
			scDescs[1]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			scDescs[1]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			scDescs[1]._predictBinary = true;
		}
		else {
			scDescs.resize(2);

			scDescs[0]._size = prevLayerSize;
			scDescs[0]._encodeRadius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			scDescs[0]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			scDescs[0]._predictBinary = true;

			scDescs[1]._size = _layerDescs[l]._size;
			scDescs[1]._encodeRadius = _layerDescs[l]._recurrentRadius;
			scDescs[1]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			scDescs[1]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			scDescs[1]._predictBinary = true;
		}

		std::vector<cl_int2> feedBackSizes(2);

		if (l < _layers.size() - 1)
			feedBackSizes[0] = feedBackSizes[1] = _layerDescs[l + 1]._size;
		else
			feedBackSizes[0] = feedBackSizes[1] = { 1, 1 };

		if (l < _layers.size() - 1)
			_layers[l]._sp.createRandom(cs, program, scDescs, _layerDescs[l]._size, feedBackSizes, _layerDescs[l]._lateralRadius, initWeightRange, rng);

		prevLayerSize = _layerDescs[l]._size;
	}

	_inputWhitener.create(cs, program, _inputSize, CL_R, CL_FLOAT);

	_zeroLayer = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 1, 1);

	cs.getQueue().enqueueFillImage(_zeroLayer, cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { 1, 1, 1 });
}

void PredictiveHierarchy::simStep(sys::ComputeSystem &cs, const cl::Image2D &input, bool learn, bool whiten) {
	// Whiten input
	if (whiten)
		_inputWhitener.filter(cs, input, _whiteningKernelRadius, _whiteningIntensity);
	
	// Feed forward
	cl::Image2D prevLayerState = whiten ? _inputWhitener.getResult() : input;

	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates(2);

			visibleStates[0] = prevLayerState;
			visibleStates[1] = _layers[l]._sp.getHiddenStates()[_back];

			_layers[l]._sp.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);

			// Get reward
			if (l < _layers.size() - 1) {
				int argIndex = 0;
				
				_predictionRewardKernel.setArg(argIndex++, _layers[l + 1]._pred.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predReward);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predRewardBaselines[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predRewardBaselines[_front]);
				_predictionRewardKernel.setArg(argIndex++, _layerDescs[l]._scActiveRatio);
				_predictionRewardKernel.setArg(argIndex++, _layerDescs[l]._predRewardBaselineDecay);

				cs.getQueue().enqueueNDRangeKernel(_predictionRewardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));

				std::swap(_layers[l]._predRewardBaselines[_front], _layers[l]._predRewardBaselines[_back]);
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

		_layers[l]._pred.activate(cs, visibleStates, l == 0 ? Predictor::_identity : Predictor::_tanH);
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
				_layers[l]._pred.learn(cs, input, visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
			else
				_layers[l]._pred.learn(cs, _layers[l - 1]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha);
		}
	}
}

void PredictiveHierarchy::clearMemory(sys::ComputeSystem &cs) {
	// Fix me
	abort();
}

void PredictiveHierarchy::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
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

		//l._sc.writeToStream(cs, os);
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

void PredictiveHierarchy::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
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

		//l._sc.readFromStream(cs, program, is);
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
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
		std::vector<SparsePredictor::VisibleLayerDesc> spDescs;

		if (l == 0) {
			spDescs.resize(2);

			spDescs[0]._size = prevLayerSize;
			spDescs[0]._encodeRadius = _layerDescs[l]._feedForwardRadius;
			spDescs[0]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			spDescs[0]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			spDescs[0]._predictThresholded = false;
			spDescs[0]._predict = true;
			spDescs[0]._ignoreMiddle = false;

			spDescs[1]._size = _layerDescs[l]._size;
			spDescs[1]._encodeRadius = _layerDescs[l]._recurrentRadius;
			spDescs[1]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			spDescs[1]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			spDescs[1]._predictThresholded = true;
			spDescs[1]._predict = false;
			spDescs[1]._ignoreMiddle = true;
		}
		else {
			spDescs.resize(2);

			spDescs[0]._size = prevLayerSize;
			spDescs[0]._encodeRadius = _layerDescs[l]._feedForwardRadius;
			spDescs[0]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			spDescs[0]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			spDescs[0]._predictThresholded = true;
			spDescs[0]._predict = true;
			spDescs[0]._ignoreMiddle = false;

			spDescs[1]._size = _layerDescs[l]._size;
			spDescs[1]._encodeRadius = _layerDescs[l]._recurrentRadius;
			spDescs[1]._predDecodeRadius = _layerDescs[l]._predictiveRadius;
			spDescs[1]._feedBackDecodeRadius = _layerDescs[l]._feedBackRadius;
			spDescs[1]._predictThresholded = true;
			spDescs[1]._predict = false;
			spDescs[1]._ignoreMiddle = true;
		}

		std::vector<cl_int2> feedBackSizes(2);

		if (l < _layers.size() - 1)
			feedBackSizes[0] = feedBackSizes[1] = _layerDescs[l]._size;
		else
			feedBackSizes[0] = feedBackSizes[1] = { 1, 1 };

		_layers[l]._sp.createRandom(cs, program, spDescs, _layerDescs[l]._size, feedBackSizes, _layerDescs[l]._lateralRadius, initWeightRange, rng);

		_layers[l]._additionalErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerSize.x, prevLayerSize.y);

		_layers[l]._hiddenStatesPrevPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cs.getQueue().enqueueFillImage(_layers[l]._additionalErrors, cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(prevLayerSize.x), static_cast<cl::size_type>(prevLayerSize.y), 1 });
		
		cs.getQueue().enqueueFillImage(_layers[l]._hiddenStatesPrevPrev, cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l]._size.x), static_cast<cl::size_type>(_layerDescs[l]._size.y), 1 });

		prevLayerSize = _layerDescs[l]._size;
	}

	_inputWhitener.create(cs, program, _inputSize, CL_R, CL_FLOAT);

	_zeroLayer = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 1, 1);

	_inputPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputSize.x, _inputSize.y);

	cs.getQueue().enqueueFillImage(_zeroLayer, cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { 1, 1, 1 });

	cs.getQueue().enqueueFillImage(_inputPrev, cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_inputSize.x), static_cast<cl::size_type>(_inputSize.y), 1 });
}

void PredictiveHierarchy::simStep(sys::ComputeSystem &cs, const cl::Image2D &input, bool learn, bool whiten) {
	// Save t - 2 state
	for (int l = 0; l < _layers.size(); l++)
		cs.getQueue().enqueueCopyImage(_layers[l]._sp.getHiddenStates()[_front], _layers[l]._hiddenStatesPrevPrev, { 0, 0, 0 }, { 0, 0, 0 }, { static_cast<cl::size_type>(_layerDescs[l]._size.x), static_cast<cl::size_type>(_layerDescs[l]._size.y), 1 });
	
	// Whiten input
	if (whiten)
		_inputWhitener.filter(cs, input, _whiteningKernelRadius, _whiteningIntensity);
	
	// Feed forward
	cl::Image2D prevLayerState = whiten ? _inputWhitener.getResult() : input;

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<cl::Image2D> visibleStates(2);

		visibleStates[0] = prevLayerState;
		visibleStates[1] = _layers[l]._sp.getHiddenStates()[_back];

		_layers[l]._sp.activateEncoder(cs, visibleStates, _layerDescs[l]._spActiveRatio);

		prevLayerState = _layers[l]._sp.getHiddenStates()[_front];
	}

	// Feed back
	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> feedBackStates(2);

		if (l < _layers.size() - 1)
			feedBackStates[0] = feedBackStates[1] = _layers[l + 1]._sp.getVisibleLayer(0)._predictions[_back];
		else
			feedBackStates[0] = feedBackStates[1] = _zeroLayer;

		_layers[l]._sp.activateDecoder(cs, feedBackStates);
	}

	if (learn) {
		// Feed forward
		prevLayerState = input;
		cl::Image2D prevLayerStatePrev = _inputPrev;

		for (int l = 0; l < _layers.size(); l++) {
			// Encoder
			std::vector<cl::Image2D> visibleStates(2);

			visibleStates[0] = prevLayerState;
			visibleStates[1] = _layers[l]._sp.getHiddenStates()[_front];

			std::vector<cl::Image2D> visibleStatesPrev(2);

			visibleStatesPrev[0] = prevLayerStatePrev;
			visibleStatesPrev[1] = _layers[l]._hiddenStatesPrevPrev;

			std::vector<cl::Image2D> feedBackStatesPrev(2);

			if (l < _layers.size() - 1)
				feedBackStatesPrev[0] = feedBackStatesPrev[1] = _layers[l + 1]._sp.getVisibleLayer(0)._predictions[_front];
			else
				feedBackStatesPrev[0] = feedBackStatesPrev[1] = _zeroLayer;

			_layers[l]._sp.learn(cs, visibleStates, visibleStatesPrev, feedBackStatesPrev, { _layers[l]._additionalErrors, _layers[l]._additionalErrors },
				_layerDescs[l]._spWeightEncodeAlpha, _layerDescs[l]._spWeightDecodeAlpha, _layerDescs[l]._spWeightLambda, _layerDescs[l]._spBiasAlpha, _layerDescs[l]._spActiveRatio, _layerDescs[l]._spImportanceDecay, _layerDescs[l]._spImportanceStrength, _layerDescs[l]._spAverageErrorDecay);

			prevLayerState = _layers[l]._sp.getHiddenStates()[_back];
			prevLayerStatePrev = _layers[l]._sp.getHiddenStates()[_front];
		}
	}

	// Keep previous (t - 1) input
	cs.getQueue().enqueueCopyImage(input, _inputPrev, { 0, 0, 0 }, { 0, 0, 0 }, { static_cast<cl::size_type>(_inputSize.x), static_cast<cl::size_type>(_inputSize.y), 1 });
}
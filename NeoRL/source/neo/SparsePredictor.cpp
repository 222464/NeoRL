#include "SparsePredictor.h"

using namespace neo;

void SparsePredictor::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, const std::vector<cl_int2> &feedBackSizes, cl_int lateralRadius, cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	_visibleLayerDescs = visibleLayerDescs;

	_hiddenSize = hiddenSize;
	_feedBackSizes = feedBackSizes;
	_lateralRadius = lateralRadius;

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	_visibleLayers.resize(_visibleLayerDescs.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
	cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

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

		vl._visibleToFeedBack = cl_float2{ static_cast<float>(_feedBackSizes[vli].x) / static_cast<float>(vld._size.x),
			static_cast<float>(_feedBackSizes[vli].y) / static_cast<float>(vld._size.y)
		};

		if (vld._useForInput) {
			int weightDiam = vld._encodeRadius * 2 + 1;

			int numWeights = weightDiam * weightDiam;

			cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

			vl._encoderWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

			randomUniform(vl._encoderWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
		}

		if (vld._predict) {
			{
				int weightDiam = vld._predDecodeRadius * 2 + 1;

				int numWeights = weightDiam * weightDiam;

				cl_int3 weightsSize = { vld._size.x, vld._size.y, numWeights };

				vl._predDecoderWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

				randomUniform(vl._predDecoderWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
			}

			{
				int weightDiam = vld._feedBackDecodeRadius * 2 + 1;

				int numWeights = weightDiam * weightDiam;

				cl_int3 weightsSize = { vld._size.x, vld._size.y, numWeights };

				vl._feedBackDecoderWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

				randomUniform(vl._feedBackDecoderWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
			}

			vl._predictions = createDoubleBuffer2D(cs, vld._size, CL_R, CL_FLOAT);

			cs.getQueue().enqueueFillImage(vl._predictions[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });

			vl._predError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);
		}
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
	_hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenActivationSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenErrorSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

	randomUniform(_hiddenBiases[_back], cs, randomUniform2DKernel, _hiddenSize, initWeightRange, rng);

	// Create kernels
	_encodeKernel = cl::Kernel(program.getProgram(), "spEncode");
	_decodeKernel = cl::Kernel(program.getProgram(), "spDecode");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "spSolveHidden");
	_predictionErrorKernel = cl::Kernel(program.getProgram(), "spPredictionError");
	_errorPropagationKernel = cl::Kernel(program.getProgram(), "spErrorPropagation");
	_learnEncoderWeightsKernel = cl::Kernel(program.getProgram(), "spLearnEncoderWeights");
	_learnDecoderWeightsKernel = cl::Kernel(program.getProgram(), "spLearnDecoderWeights");
	_learnBiasesKernel = cl::Kernel(program.getProgram(), "spLearnBiases");
}

void SparsePredictor::activateEncoder(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio) {
	// Start by clearing activation summation buffer
	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueCopyImage(_hiddenBiases[_back], _hiddenActivationSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);
	}

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		if (vld._useForInput) {
			int argIndex = 0;

			_encodeKernel.setArg(argIndex++, visibleStates[vli]);
			_encodeKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
			_encodeKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_front]);
			_encodeKernel.setArg(argIndex++, vl._encoderWeights[_back]);
			_encodeKernel.setArg(argIndex++, vld._size);
			_encodeKernel.setArg(argIndex++, vl._hiddenToVisible);
			_encodeKernel.setArg(argIndex++, vld._encodeRadius);
			_encodeKernel.setArg(argIndex++, vld._ignoreMiddle);

			cs.getQueue().enqueueNDRangeKernel(_encodeKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			// Swap buffers
			std::swap(_hiddenActivationSummationTemp[_front], _hiddenActivationSummationTemp[_back]);
		}
	}

	{
		int argIndex = 0;

		_solveHiddenKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenSize);
		_solveHiddenKernel.setArg(argIndex++, _lateralRadius);
		_solveHiddenKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}
	
	// No buffer swapping yet, this happens in the decoding phase
}

void SparsePredictor::activateDecoder(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &feedBackStates) {
	// Now decode
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		if (vld._predict) {
			int argIndex = 0;

			_decodeKernel.setArg(argIndex++, _hiddenStates[_front]);
			_decodeKernel.setArg(argIndex++, feedBackStates[vli]);
			_decodeKernel.setArg(argIndex++, vl._predictions[_front]);
			_decodeKernel.setArg(argIndex++, vl._predDecoderWeights[_back]);
			_decodeKernel.setArg(argIndex++, vl._feedBackDecoderWeights[_back]);
			_decodeKernel.setArg(argIndex++, _hiddenSize);
			_decodeKernel.setArg(argIndex++, _feedBackSizes[vli]);
			_decodeKernel.setArg(argIndex++, vl._visibleToHidden);
			_decodeKernel.setArg(argIndex++, vl._visibleToFeedBack);
			_decodeKernel.setArg(argIndex++, vld._predDecodeRadius);
			_decodeKernel.setArg(argIndex++, vld._feedBackDecodeRadius);
			_decodeKernel.setArg(argIndex++, vld._predictThresholded);

			cs.getQueue().enqueueNDRangeKernel(_decodeKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
		}
	}

	// Swap buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		if (vld._predict)
			std::swap(vl._predictions[_front], vl._predictions[_back]);
	}
}

void SparsePredictor::learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates,
	const std::vector<cl::Image2D> &feedBackStatesPrev, const std::vector<cl::Image2D> &addidionalErrors, float weightEncodeAlpha, float weightDecodeAlpha, float weightLambda, float biasAlpha, float activeRatio)
{
	// Start by clearing error summation buffer
	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_hiddenErrorSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);
	}

	// Find error
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		if (vld._predict) {
			{
				int argIndex = 0;

				_predictionErrorKernel.setArg(argIndex++, vl._predictions[_front]);
				_predictionErrorKernel.setArg(argIndex++, visibleStates[vli]);
				_predictionErrorKernel.setArg(argIndex++, addidionalErrors[vli]);
				_predictionErrorKernel.setArg(argIndex++, vl._predError);

				cs.getQueue().enqueueNDRangeKernel(_predictionErrorKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
			}

			// Propagate the error
			{
				cl_int2 reversePredDecodeRadii = { static_cast<int>(std::ceil(vl._visibleToHidden.x * (vld._predDecodeRadius + 0.5f))), static_cast<int>(std::ceil(vl._visibleToHidden.y * (vld._predDecodeRadius + 0.5f))) };

				int argIndex = 0;

				_errorPropagationKernel.setArg(argIndex++, vl._predError);
				_errorPropagationKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
				_errorPropagationKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_front]);
				_errorPropagationKernel.setArg(argIndex++, vl._predDecoderWeights[_back]);
				_errorPropagationKernel.setArg(argIndex++, vld._size);
				_errorPropagationKernel.setArg(argIndex++, _hiddenSize);
				_errorPropagationKernel.setArg(argIndex++, vl._visibleToHidden);
				_errorPropagationKernel.setArg(argIndex++, vl._hiddenToVisible);
				_errorPropagationKernel.setArg(argIndex++, vld._predDecodeRadius);
				_errorPropagationKernel.setArg(argIndex++, reversePredDecodeRadii);

				cs.getQueue().enqueueNDRangeKernel(_errorPropagationKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
			}

			std::swap(_hiddenErrorSummationTemp[_front], _hiddenErrorSummationTemp[_back]);
		}
	}
	
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		// Decoder
		if (vld._predict) {
			int argIndex = 0;

			_learnDecoderWeightsKernel.setArg(argIndex++, vl._predError);
			_learnDecoderWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
			_learnDecoderWeightsKernel.setArg(argIndex++, feedBackStatesPrev[vli]);
			_learnDecoderWeightsKernel.setArg(argIndex++, vl._predDecoderWeights[_back]);
			_learnDecoderWeightsKernel.setArg(argIndex++, vl._predDecoderWeights[_front]);
			_learnDecoderWeightsKernel.setArg(argIndex++, vl._feedBackDecoderWeights[_back]);
			_learnDecoderWeightsKernel.setArg(argIndex++, vl._feedBackDecoderWeights[_front]);
			_learnDecoderWeightsKernel.setArg(argIndex++, _hiddenSize);
			_learnDecoderWeightsKernel.setArg(argIndex++, _feedBackSizes[vli]);
			_learnDecoderWeightsKernel.setArg(argIndex++, vl._visibleToHidden);
			_learnDecoderWeightsKernel.setArg(argIndex++, vl._visibleToFeedBack);
			_learnDecoderWeightsKernel.setArg(argIndex++, vld._predDecodeRadius);
			_learnDecoderWeightsKernel.setArg(argIndex++, vld._feedBackDecodeRadius);
			_learnDecoderWeightsKernel.setArg(argIndex++, weightDecodeAlpha);

			cs.getQueue().enqueueNDRangeKernel(_learnDecoderWeightsKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));

			std::swap(vl._predDecoderWeights[_front], vl._predDecoderWeights[_back]);
			std::swap(vl._feedBackDecoderWeights[_front], vl._feedBackDecoderWeights[_back]);
		}

		// Encoder
		if (vld._useForInput) {
			int argIndex = 0;

			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, visibleStates[vli]);
			_learnEncoderWeightsKernel.setArg(argIndex++, vl._encoderWeights[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, vl._encoderWeights[_front]);
			_learnEncoderWeightsKernel.setArg(argIndex++, vld._size);
			_learnEncoderWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnEncoderWeightsKernel.setArg(argIndex++, vld._encodeRadius);
			_learnEncoderWeightsKernel.setArg(argIndex++, weightEncodeAlpha);
			_learnEncoderWeightsKernel.setArg(argIndex++, weightLambda);

			cs.getQueue().enqueueNDRangeKernel(_learnEncoderWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			std::swap(vl._encoderWeights[_front], vl._encoderWeights[_back]);
		}
	}

	// Biases
	{
		int argIndex = 0;

		_learnBiasesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
		_learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
		_learnBiasesKernel.setArg(argIndex++, biasAlpha);
		_learnBiasesKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
	}
}
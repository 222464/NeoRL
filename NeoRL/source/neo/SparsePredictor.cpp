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

		{
			int weightDiam = vld._encodeRadius * 2 + 1;

			int numWeights = weightDiam * weightDiam;

			cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

			vl._encoderWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

			randomUniform(vl._encoderWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
		}

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
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenActivationSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenErrorSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_encodeKernel = cl::Kernel(program.getProgram(), "spEncode");
	_decodeKernel = cl::Kernel(program.getProgram(), "spDecode");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "spSolveHidden");
	_predictionErrorKernel = cl::Kernel(program.getProgram(), "spPredictionError");
	_errorPropagationKernel = cl::Kernel(program.getProgram(), "spErrorPropagation");
	_learnEncoderWeightsKernel = cl::Kernel(program.getProgram(), "spLearnEncoderWeights");
	_learnDecoderWeightsKernel = cl::Kernel(program.getProgram(), "spLearnDecoderWeights");
}

void SparsePredictor::activateEncoder(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio) {
	// Start by clearing activation summation buffer
	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_hiddenActivationSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);
	}

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_encodeKernel.setArg(argIndex++, visibleStates[vli]);
		_encodeKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
		_encodeKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_front]);
		_encodeKernel.setArg(argIndex++, vl._encoderWeights[_back]);
		_encodeKernel.setArg(argIndex++, vld._size);
		_encodeKernel.setArg(argIndex++, vl._hiddenToVisible);
		_encodeKernel.setArg(argIndex++, vld._encodeRadius);

		cs.getQueue().enqueueNDRangeKernel(_encodeKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		// Swap buffers
		std::swap(_hiddenActivationSummationTemp[_front], _hiddenActivationSummationTemp[_back]);
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
	
	// No buffer swapping yet
}

void SparsePredictor::activateDecoder(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &feedBackStates) {
	// Start by clearing activation summation buffer
	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_hiddenActivationSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);
	}

	// Now decode
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

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
		_decodeKernel.setArg(argIndex++, vld._predictBinary);

		cs.getQueue().enqueueNDRangeKernel(_decodeKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}

	// Swap buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		std::swap(vl._predictions[_front], vl._predictions[_back]);
	}
}

void SparsePredictor::learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates,
	const std::vector<cl::Image2D> &feedBackStates, const std::vector<cl::Image2D> &addidionalErrors, float weightAlpha, float weightLambda)
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

		{
			int argIndex = 0;

			_predictionErrorKernel.setArg(argIndex++, vl._predictions[_front]);
			_predictionErrorKernel.setArg(argIndex++, visibleStates[vli]);
			_predictionErrorKernel.setArg(argIndex++, addidionalErrors[vli]);
			_predictionErrorKernel.setArg(argIndex++, vl._error);

			cs.getQueue().enqueueNDRangeKernel(_predictionErrorKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
		}

		// Propagate the error
		{
			cl_int2 reversePredDecodeRadii = { static_cast<int>(std::ceil(vl._visibleToHidden.x * (vld._predDecodeRadius + 0.5f))), static_cast<int>(std::ceil(vl._visibleToHidden.y * (vld._predDecodeRadius + 0.5f))) };
			
			int argIndex = 0;

			_errorPropagationKernel.setArg(argIndex++, vl._error);
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
	
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		// Decoder
		{
			int argIndex = 0;

			_learnDecoderWeightsKernel.setArg(argIndex++, vl._error);
			_learnDecoderWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
			_learnDecoderWeightsKernel.setArg(argIndex++, feedBackStates[vli]);
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
			_learnDecoderWeightsKernel.setArg(argIndex++, weightAlpha);

			cs.getQueue().enqueueNDRangeKernel(_learnDecoderWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			std::swap(vl._predDecoderWeights[_front], vl._predDecoderWeights[_back]);
		}

		// Encoder
		{
			int argIndex = 0;

			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
			_learnEncoderWeightsKernel.setArg(argIndex++, visibleStates[vli]);
			_learnEncoderWeightsKernel.setArg(argIndex++, vl._encoderWeights[_back]);
			_learnEncoderWeightsKernel.setArg(argIndex++, vl._encoderWeights[_front]);
			_learnEncoderWeightsKernel.setArg(argIndex++, vld._size);
			_learnEncoderWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnEncoderWeightsKernel.setArg(argIndex++, vld._encodeRadius);
			_learnEncoderWeightsKernel.setArg(argIndex++, weightAlpha);

			cs.getQueue().enqueueNDRangeKernel(_learnEncoderWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			std::swap(vl._encoderWeights[_front], vl._encoderWeights[_back]);
		}
	}
}
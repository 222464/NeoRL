#include "SparseCoder.h"

using namespace neo;

void SparseCoder::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
	bool enableTraces,
	std::mt19937 &rng)
{
	const cl_channel_order weightChannels = enableTraces ? CL_RG : CL_R;

	_visibleLayerDescs = visibleLayerDescs;

	_hiddenSize = hiddenSize;

	_lateralRadius = lateralRadius;

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl_float4 thresholdColor = { initThreshold, initThreshold, initThreshold, initThreshold };

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

		vl._reverseRadii = cl_int2 { static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius)), static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius)) };

		// Create images
		vl._reconstructionError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		cs.getQueue().enqueueFillImage(vl._reconstructionError, zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
		
		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3{ _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, weightChannels, CL_FLOAT);

		randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
	_hiddenSpikes = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
	_hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenThresholds = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	{
		int lateralWeightDiam = lateralRadius * 2 + 1;

		int numLateralWeights = lateralWeightDiam * lateralWeightDiam;

		cl_int3 lateralWeightsSize = cl_int3 { _hiddenSize.x, _hiddenSize.y, numLateralWeights };

		_lateralWeights = createDoubleBuffer3D(cs, lateralWeightsSize, CL_R, CL_FLOAT);
	
		randomUniform(_lateralWeights[_back], cs, randomUniform3DKernel, lateralWeightsSize, initLateralWeightRange, rng);
	}

	cs.getQueue().enqueueFillImage(_hiddenThresholds[_back], thresholdColor, zeroOrigin, hiddenRegion);

	cs.getQueue().enqueueFillImage(_hiddenSpikes[_back], zeroColor, zeroOrigin, hiddenRegion);
	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
	cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_reconstructVisibleErrorKernel = cl::Kernel(program.getProgram(), "scReconstructVisibleError");
	_activateFromReconstructionErrorKernel = cl::Kernel(program.getProgram(), "scActivateFromReconstructionError");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "scSolveHidden");
	_learnThresholdsKernel = cl::Kernel(program.getProgram(), "scLearnThresholds");
	_learnWeightsKernel = cl::Kernel(program.getProgram(), "scLearnSparseCoderWeights");
	_learnWeightsTracesKernel = cl::Kernel(program.getProgram(), "scLearnSparseCoderWeightsTraces");
	_learnWeightsLateralKernel = cl::Kernel(program.getProgram(), "scLearnSparseCoderWeightsLateral");
}

void SparseCoder::reconstructError(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates) {
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_reconstructVisibleErrorKernel.setArg(argIndex++, _hiddenStates[_back]);
		_reconstructVisibleErrorKernel.setArg(argIndex++, visibleStates[vli]);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vl._reconstructionError);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vl._weights[_back]);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vld._size);
		_reconstructVisibleErrorKernel.setArg(argIndex++, _hiddenSize);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vl._visibleToHidden);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vl._hiddenToVisible);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vld._radius);
		_reconstructVisibleErrorKernel.setArg(argIndex++, vl._reverseRadii);

		cs.getQueue().enqueueNDRangeKernel(_reconstructVisibleErrorKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}
}

void SparseCoder::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, cl_int iterations, cl_float leak) {
	// Clear previous aggregate state information
	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
		cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);
	}

	for (cl_int iter = 0; iter < iterations; iter++) {
		// Start by clearing summation buffer
		{
			cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

			cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

			cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);
		}

		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
			VisibleLayer &vl = _visibleLayers[vli];
			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

			int argIndex = 0;

			_activateFromReconstructionErrorKernel.setArg(argIndex++, vl._reconstructionError);
			_activateFromReconstructionErrorKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_activateFromReconstructionErrorKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
			_activateFromReconstructionErrorKernel.setArg(argIndex++, vl._weights[_back]);
			_activateFromReconstructionErrorKernel.setArg(argIndex++, vld._size);
			_activateFromReconstructionErrorKernel.setArg(argIndex++, vl._hiddenToVisible);
			_activateFromReconstructionErrorKernel.setArg(argIndex++, vld._radius);

			cs.getQueue().enqueueNDRangeKernel(_activateFromReconstructionErrorKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			// Swap buffers
			std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
		}

		// Back now contains the sums. Solve sparse codes from this
		{
			int argIndex = 0;

			_solveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenSpikes[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenSpikes[_front]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenActivations[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenActivations[_front]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenThresholds[_back]);
			_solveHiddenKernel.setArg(argIndex++, _lateralWeights[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenSize);
			_solveHiddenKernel.setArg(argIndex++, _lateralRadius);
			_solveHiddenKernel.setArg(argIndex++, leak);
			_solveHiddenKernel.setArg(argIndex++, 1.0f / (1.0f + iter));

			cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		// Swap hidden state buffers
		std::swap(_hiddenSpikes[_front], _hiddenSpikes[_back]);
		std::swap(_hiddenStates[_front], _hiddenStates[_back]);
		std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);

		reconstructError(cs, visibleStates);
	}
}

void SparseCoder::learn(sys::ComputeSystem &cs, float weightAlpha, float weightLateralAlpha, float thresholdAlpha, float activeRatio) {
	// Learn Thresholds
	{
		int argIndex = 0;

		_learnThresholdsKernel.setArg(argIndex++, _hiddenThresholds[_back]);
		_learnThresholdsKernel.setArg(argIndex++, _hiddenThresholds[_front]);
		_learnThresholdsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnThresholdsKernel.setArg(argIndex++, thresholdAlpha);
		_learnThresholdsKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnThresholdsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenThresholds[_front], _hiddenThresholds[_back]);
	}

	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsKernel.setArg(argIndex++, vl._reconstructionError);
		_learnWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsKernel.setArg(argIndex++, vld._size);
		_learnWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsKernel.setArg(argIndex++, vld._radius);
		_learnWeightsKernel.setArg(argIndex++, weightAlpha);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}

	// Learn lateral weights
	{
		int argIndex = 0;

		_learnWeightsLateralKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsLateralKernel.setArg(argIndex++, _lateralWeights[_back]);
		_learnWeightsLateralKernel.setArg(argIndex++, _lateralWeights[_front]);
		_learnWeightsLateralKernel.setArg(argIndex++, _hiddenSize);
		_learnWeightsLateralKernel.setArg(argIndex++, _lateralRadius);
		_learnWeightsLateralKernel.setArg(argIndex++, weightLateralAlpha);
		_learnWeightsLateralKernel.setArg(argIndex++, activeRatio * activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsLateralKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_lateralWeights[_front], _lateralWeights[_back]);
	}
}

void SparseCoder::learnTrace(sys::ComputeSystem &cs, const cl::Image2D &rewards, float weightAlpha, float weightLateralAlpha, float weightTraceLambda, float thresholdAlpha, float activeRatio) {
	// Learn Thresholds
	{
		int argIndex = 0;

		_learnThresholdsKernel.setArg(argIndex++, _hiddenThresholds[_back]);
		_learnThresholdsKernel.setArg(argIndex++, _hiddenThresholds[_front]);
		_learnThresholdsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnThresholdsKernel.setArg(argIndex++, thresholdAlpha);
		_learnThresholdsKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnThresholdsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenThresholds[_front], _hiddenThresholds[_back]);
	}

	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsTracesKernel.setArg(argIndex++, vl._reconstructionError);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsTracesKernel.setArg(argIndex++, rewards);
		_learnWeightsTracesKernel.setArg(argIndex++, vld._size);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsTracesKernel.setArg(argIndex++, vld._radius);
		_learnWeightsTracesKernel.setArg(argIndex++, weightAlpha);
		_learnWeightsTracesKernel.setArg(argIndex++, weightTraceLambda);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}

	// Learn lateral weights
	{
		int argIndex = 0;

		_learnWeightsLateralKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsLateralKernel.setArg(argIndex++, _lateralWeights[_back]);
		_learnWeightsLateralKernel.setArg(argIndex++, _lateralWeights[_front]);
		_learnWeightsLateralKernel.setArg(argIndex++, _hiddenSize);
		_learnWeightsLateralKernel.setArg(argIndex++, _lateralRadius);
		_learnWeightsLateralKernel.setArg(argIndex++, weightLateralAlpha);
		_learnWeightsLateralKernel.setArg(argIndex++, activeRatio * activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsLateralKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_lateralWeights[_front], _lateralWeights[_back]);
	}
}
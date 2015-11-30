#include "ComparisonSparseCoder.h"

using namespace neo;

void ComparisonSparseCoder::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs,
	cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float initThreshold,
	bool enableTraces,
	std::mt19937 &rng)
{
	const cl_channel_order weightChannels = enableTraces ? CL_RG : CL_R;

	_visibleLayerDescs = visibleLayerDescs;

	_lateralRadius = lateralRadius;

	_hiddenSize = hiddenSize;

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

		vl._reverseRadii = cl_int2{ static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius)), static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius)) };

		// Create images
		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3{ _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, weightChannels, CL_FLOAT);

		randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenThresholds = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl_float4 thresholdColor = { initThreshold, initThreshold, initThreshold, initThreshold };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	cs.getQueue().enqueueFillImage(_hiddenThresholds[_back], thresholdColor, zeroOrigin, hiddenRegion);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_activateKernel = cl::Kernel(program.getProgram(), "cscActivate");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "cscSolveHidden");
	_learnThresholdsKernel = cl::Kernel(program.getProgram(), "cscLearnThresholds");
	_learnWeightsKernel = cl::Kernel(program.getProgram(), "cscLearnWeights");
	_learnWeightsTracesKernel = cl::Kernel(program.getProgram(), "cscLearnWeightsTraces");
}

void ComparisonSparseCoder::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio) {
	// Start by clearing summation buffer to thresholds
	{
		//cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueCopyImage(_hiddenThresholds[_back], _hiddenSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);
		//cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);
	}

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_activateKernel.setArg(argIndex++, visibleStates[vli]);
		_activateKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_activateKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
		_activateKernel.setArg(argIndex++, vl._weights[_back]);
		_activateKernel.setArg(argIndex++, vld._size);
		_activateKernel.setArg(argIndex++, vl._hiddenToVisible);
		_activateKernel.setArg(argIndex++, vld._radius);

		cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		// Swap buffers
		std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
	}

	// Back now contains the sums. Solve sparse codes from this
	{
		int argIndex = 0;

		_solveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenSize);
		_solveHiddenKernel.setArg(argIndex++, _lateralRadius);
		_solveHiddenKernel.setArg(argIndex++, activeRatio);
		
		cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Swap hidden state buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);
}

void ComparisonSparseCoder::learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float weightAlpha, float thresholdAlpha, float activeRatio) {
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

		_learnWeightsKernel.setArg(argIndex++, visibleStates[vli]);
		_learnWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsKernel.setArg(argIndex++, vld._size);
		_learnWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsKernel.setArg(argIndex++, vld._radius);
		_learnWeightsKernel.setArg(argIndex++, weightAlpha);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void ComparisonSparseCoder::learnTrace(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &rewards, float weightAlpha, float weightTraceLambda, float thresholdAlpha, float activeRatio) {
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

		_learnWeightsTracesKernel.setArg(argIndex++, visibleStates[vli]);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
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
}
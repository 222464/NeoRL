#include "PredictorSwarm.h"

using namespace neo;

void PredictorSwarm::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_visibleLayerDescs = visibleLayerDescs;

	_hiddenSize = hiddenSize;

	_visibleLayers.resize(_visibleLayerDescs.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
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

		vl._reverseRadii = cl_int2{ static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius)), static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius)) };

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_RGBA, CL_FLOAT);

		randomUniformXZ(vl._weights[_back], cs, randomUniform3DXZKernel, weightsSize, initWeightRange, rng);
	}

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	_hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);
	
	_inhibitionTemp = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _hiddenSize.x, _hiddenSize.y);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
	cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_activateKernel = cl::Kernel(program.getProgram(), "predActivateSwarm");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "predSolveHiddenSwarm");
	_solveHiddenModulatedKernel = cl::Kernel(program.getProgram(), "predSolveHiddenModulatedSwarm");
	_learnWeightsTracesKernel = cl::Kernel(program.getProgram(), "predLearnWeightsTracesSwarm");
	_learnWeightsTracesModulatedKernel = cl::Kernel(program.getProgram(), "predLearnWeightsTracesModulatedSwarm");
}

void PredictorSwarm::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float noise, std::mt19937 &rng) {
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

	std::uniform_int_distribution<int> seedDist(0, 999);

	cl_uint2 seed = { seedDist(rng), seedDist(rng) };

	{
		int argIndex = 0;

		_solveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenActivations[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenActivations[_front]);
		_solveHiddenKernel.setArg(argIndex++, noise);
		_solveHiddenKernel.setArg(argIndex++, seed);

		cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Swap hidden state buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);
	std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
}

void PredictorSwarm::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &modulatorImage, float noise, std::mt19937 &rng) {
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

	std::uniform_int_distribution<int> seedDist(0, 999);

	cl_uint2 seed = { seedDist(rng), seedDist(rng) };

	{
		int argIndex = 0;

		_solveHiddenModulatedKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_solveHiddenModulatedKernel.setArg(argIndex++, modulatorImage);
		_solveHiddenModulatedKernel.setArg(argIndex++, _hiddenStates[_back]);
		_solveHiddenModulatedKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenModulatedKernel.setArg(argIndex++, _hiddenActivations[_back]);
		_solveHiddenModulatedKernel.setArg(argIndex++, _hiddenActivations[_front]);
		_solveHiddenModulatedKernel.setArg(argIndex++, noise);
		_solveHiddenModulatedKernel.setArg(argIndex++, seed);

		cs.getQueue().enqueueNDRangeKernel(_solveHiddenModulatedKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Swap hidden state buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);
	std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
}

void PredictorSwarm::learn(sys::ComputeSystem &cs, float reward, float gamma, std::vector<cl::Image2D> &visibleStatesPrev, cl_float2 weightAlpha, cl_float2 weightLambda) {
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsTracesKernel.setArg(argIndex++, visibleStatesPrev[vli]);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenActivations[_front]);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_front]);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsTracesKernel.setArg(argIndex++, vld._size);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsTracesKernel.setArg(argIndex++, vld._radius);
		_learnWeightsTracesKernel.setArg(argIndex++, weightAlpha);
		_learnWeightsTracesKernel.setArg(argIndex++, weightLambda);
		_learnWeightsTracesKernel.setArg(argIndex++, reward);
		_learnWeightsTracesKernel.setArg(argIndex++, gamma);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void PredictorSwarm::learn(sys::ComputeSystem &cs, float reward, float gamma, std::vector<cl::Image2D> &visibleStatesPrev, const cl::Image2D &modulatorImage, cl_float2 weightAlpha, cl_float2 weightLambda) {
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsTracesModulatedKernel.setArg(argIndex++, visibleStatesPrev[vli]);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, modulatorImage);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, _hiddenActivations[_front]);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, _hiddenStates[_front]);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, vld._size);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, vld._radius);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, weightAlpha);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, weightLambda);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, reward);
		_learnWeightsTracesModulatedKernel.setArg(argIndex++, gamma);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsTracesModulatedKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}
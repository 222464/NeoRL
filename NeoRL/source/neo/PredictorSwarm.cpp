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

		vl._reverseRadii = cl_int2{ static_cast<int>(std::ceil(vl._visibleToHidden.x * (vld._radius + 0.5f))), static_cast<int>(std::ceil(vl._visibleToHidden.y * (vld._radius + 0.5f))) };

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_RGBA, CL_FLOAT);

		randomUniformXZ(vl._weights[_back], cs, randomUniform3DXZKernel, weightsSize, initWeightRange, rng);

		// Create images
		vl._reconstructionError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);
	}

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	_hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	_hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_hiddenBiases[_back], zeroColor, zeroOrigin, hiddenRegion);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
	cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_activateKernel = cl::Kernel(program.getProgram(), "predActivateSwarm");
	_inhibitKernel = cl::Kernel(program.getProgram(), "predInhibitSwarm");
	_learnBiasesKernel = cl::Kernel(program.getProgram(), "predLearnBiasesSwarm");
	_learnWeightsTracesInhibitedKernel = cl::Kernel(program.getProgram(), "predLearnWeightsTracesSwarm");
	_reconstructionErrorKernel = cl::Kernel(program.getProgram(), "predReconstructionErrorSwarm");
}

void PredictorSwarm::activate(sys::ComputeSystem &cs, const cl::Image2D &targets, const std::vector<cl::Image2D> &visibleStates, const std::vector<cl::Image2D> &visibleStatesPrev, float activeRatio, int inhibitionRadius, std::mt19937 &rng) {
	// Start by clearing summation buffer
	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		//cs.getQueue().enqueueCopyImage(_hiddenBiases[_back], _hiddenSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);
		cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);
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

	{
		int argIndex = 0;

		_inhibitKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_inhibitKernel.setArg(argIndex++, _hiddenStates[_front]);
		_inhibitKernel.setArg(argIndex++, _hiddenSize);
		_inhibitKernel.setArg(argIndex++, inhibitionRadius);
		_inhibitKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Reconstruction error
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_reconstructionErrorKernel.setArg(argIndex++, targets);
		_reconstructionErrorKernel.setArg(argIndex++, visibleStatesPrev[vli]);
		_reconstructionErrorKernel.setArg(argIndex++, vl._reconstructionError);
		_reconstructionErrorKernel.setArg(argIndex++, vl._weights[_back]);
		_reconstructionErrorKernel.setArg(argIndex++, vld._size);
		_reconstructionErrorKernel.setArg(argIndex++, _hiddenSize);
		_reconstructionErrorKernel.setArg(argIndex++, vl._visibleToHidden);
		_reconstructionErrorKernel.setArg(argIndex++, vl._hiddenToVisible);
		_reconstructionErrorKernel.setArg(argIndex++, vld._radius);
		_reconstructionErrorKernel.setArg(argIndex++, vl._reverseRadii);

		cs.getQueue().enqueueNDRangeKernel(_reconstructionErrorKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}

	// Swap hidden state buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);
	std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
}

void PredictorSwarm::learn(sys::ComputeSystem &cs, float reward, float gamma, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, cl_float2 weightAlpha, cl_float2 weightLambda, cl_float biasAlpha, cl_float activeRatio) {
	{
		// Learn biases
		{
			int argIndex = 0;

			_learnBiasesKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
			_learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
			_learnBiasesKernel.setArg(argIndex++, biasAlpha);
			_learnBiasesKernel.setArg(argIndex++, activeRatio);

			cs.getQueue().enqueueNDRangeKernel(_learnBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		// Learn weights
		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
			VisibleLayer &vl = _visibleLayers[vli];
			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

			int argIndex = 0;

			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, visibleStatesPrev[vli]);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, vl._reconstructionError);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, targets);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, _hiddenActivations[_front]);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, _hiddenStates[_front]);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, vl._weights[_back]);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, vl._weights[_front]);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, vld._size);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, vld._radius);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, weightAlpha);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, weightLambda);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, reward);
			_learnWeightsTracesInhibitedKernel.setArg(argIndex++, gamma);

			cs.getQueue().enqueueNDRangeKernel(_learnWeightsTracesInhibitedKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			std::swap(vl._weights[_front], vl._weights[_back]);
		}
	}
}
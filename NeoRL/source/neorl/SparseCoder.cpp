#include "SparseCoder.h"

using namespace neo;

void SparseCoder::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange, cl_float initBoost,
	std::mt19937 &rng)
{
	_visibleLayerDescs = visibleLayerDescs;

	_hiddenSize = hiddenSize;

	_visibleLayers.resize(_visibleLayerDescs.size());

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

		vl._reverseRadii = cl_int2{ std::ceil(vl._visibleToHidden.x * vld._size.x), std::ceil(vl._visibleToHidden.y * vld._size.y) };

		// Create images
		vl._reconstructionTemp = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3{ _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize);

		randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize);
	_hiddenBoosts = createDoubleBuffer2D(cs, _hiddenSize);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize);

	cl_float4 zeroColor = { 0, 0, 0, 0 };
	cl_float4 boostColor = { initBoost, initBoost, initBoost, initBoost };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
	cs.getQueue().enqueueFillImage(_hiddenBoosts[_back], boostColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_reconstructVisibleKernel = cl::Kernel(program.getProgram(), "reconstructVisible");
	_reconstructionErrorKernel = cl::Kernel(program.getProgram(), "reconstructionError");
	_activateFromReconstructionKernel = cl::Kernel(program.getProgram(), "activateFromReconstruction");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "solveHidden");
	_learnBoostsKernel = cl::Kernel(program.getProgram(), "learnBoosts");
	_learnWeightsKernel = cl::Kernel(program.getProgram(), "learnSparseCoderWeights");
}

void SparseCoder::reconstruct(sys::ComputeSystem &cs) {
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_reconstructVisibleKernel.setArg(argIndex++, _hiddenStates[_back]);
		_reconstructVisibleKernel.setArg(argIndex++, vl._reconstructionTemp);
		_reconstructVisibleKernel.setArg(argIndex++, vld._size);
		_reconstructVisibleKernel.setArg(argIndex++, _hiddenSize);
		_reconstructVisibleKernel.setArg(argIndex++, vl._visibleToHidden);
		_reconstructVisibleKernel.setArg(argIndex++, vl._hiddenToVisible);
		_reconstructVisibleKernel.setArg(argIndex++, vl._reverseRadii);

		cs.getQueue().enqueueNDRangeKernel(_reconstructVisibleKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}
}

void SparseCoder::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, int iterations, float stepSize) {
	for (int iter = 0; iter < iterations; iter++) {
		reconstruct(cs);

		// Start by clearing summation buffer
		cl_float4 zeroColor = { 0, 0, 0, 0 };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);

		for (int vli = 0; vli < _visibleLayers.size(); vli++) {
			VisibleLayer &vl = _visibleLayers[vli];
			VisibleLayerDesc &vld = _visibleLayerDescs[vli];

			int argIndex = 0;

			_activateFromReconstructionKernel.setArg(argIndex++, vl._reconstructionTemp);
			_activateFromReconstructionKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_activateFromReconstructionKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
			_activateFromReconstructionKernel.setArg(argIndex++, vld._size);
			_activateFromReconstructionKernel.setArg(argIndex++, vl._hiddenToVisible);
			_activateFromReconstructionKernel.setArg(argIndex++, vld._radius);

			cs.getQueue().enqueueNDRangeKernel(_activateFromReconstructionKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

			// Swap buffers
			std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
		}

		// Back now contains the sums. Solve sparse codes from this
		{
			int argIndex = 0;

			_solveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
			_solveHiddenKernel.setArg(argIndex++, _hiddenBoosts[_back]);
			_solveHiddenKernel.setArg(argIndex++, stepSize);

			cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		// Swap hidden state buffers
		std::swap(_hiddenStates[_front], _hiddenStates[_back]);
	}

	// Final reconstruction
	reconstruct(cs);
}

void SparseCoder::learn(sys::ComputeSystem &cs, float weightAlpha, float boostAlpha, float activeRatio) {
	// Learn boosts
	{
		int argIndex = 0;

		_learnBoostsKernel.setArg(argIndex++, _hiddenBoosts[_back]);
		_learnBoostsKernel.setArg(argIndex++, _hiddenBoosts[_front]);
		_learnBoostsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnBoostsKernel.setArg(argIndex++, boostAlpha);
		_learnBoostsKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnBoostsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenBoosts[_front], _hiddenBoosts[_back]);
	}

	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsKernel.setArg(argIndex++, vl._reconstructionTemp);
		_learnWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsKernel.setArg(argIndex++, vld._size);
		_learnWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsKernel.setArg(argIndex++, vld._radius);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}
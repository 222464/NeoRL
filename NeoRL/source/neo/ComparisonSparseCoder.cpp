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
		vl._reconstructionError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3{ _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, weightChannels, CL_FLOAT);

		randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	randomUniform(_hiddenBiases[_back], cs, randomUniform2DKernel, _hiddenSize, initWeightRange, rng);

	_hiddenActivationSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
	_hiddenErrorSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_forwardErrorKernel = cl::Kernel(program.getProgram(), "cscForwardError");
	_activateKernel = cl::Kernel(program.getProgram(), "cscActivate");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "cscSolveHidden");
	_learnHiddenBiasesKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenBiases");
	_learnHiddenBiasesTracesKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenBiasesTraces");
	_learnHiddenWeightsKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenWeights");
	_learnHiddenWeightsTracesKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenWeightsTraces");
}

void ComparisonSparseCoder::reconstructError(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates) {
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_forwardErrorKernel.setArg(argIndex++, _hiddenStates[_back]);
		_forwardErrorKernel.setArg(argIndex++, visibleStates[vli]);
		_forwardErrorKernel.setArg(argIndex++, vl._reconstructionError);
		_forwardErrorKernel.setArg(argIndex++, vl._weights[_back]);
		_forwardErrorKernel.setArg(argIndex++, vld._size);
		_forwardErrorKernel.setArg(argIndex++, _hiddenSize);
		_forwardErrorKernel.setArg(argIndex++, vl._visibleToHidden);
		_forwardErrorKernel.setArg(argIndex++, vl._hiddenToVisible);
		_forwardErrorKernel.setArg(argIndex++, vld._radius);
		_forwardErrorKernel.setArg(argIndex++, vl._reverseRadii);

		cs.getQueue().enqueueNDRangeKernel(_forwardErrorKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}
}

void ComparisonSparseCoder::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio) {
	// Start by clearing summation buffer to biases
	{
		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueCopyImage(_hiddenBiases[_back], _hiddenActivationSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);
	}

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_activateKernel.setArg(argIndex++, visibleStates[vli]);
		_activateKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
		_activateKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_front]);
		_activateKernel.setArg(argIndex++, vl._weights[_back]);
		_activateKernel.setArg(argIndex++, vld._size);
		_activateKernel.setArg(argIndex++, vl._hiddenToVisible);
		_activateKernel.setArg(argIndex++, vld._radius);

		cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		// Swap buffers
		std::swap(_hiddenActivationSummationTemp[_front], _hiddenActivationSummationTemp[_back]);
	}

	// Back now contains the sums. Solve sparse codes from this
	{
		int argIndex = 0;

		_solveHiddenKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenSize);
		_solveHiddenKernel.setArg(argIndex++, _lateralRadius);
		_solveHiddenKernel.setArg(argIndex++, activeRatio);
		
		cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Swap hidden state buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);

	// Reconstruct (second layer forward + error step)
	reconstructError(cs, visibleStates);

	// Backpropagation - start by clearing summation buffer to zero
	{
		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

		cs.getQueue().enqueueFillImage(_hiddenErrorSummationTemp[_back], zeroColor, zeroOrigin, hiddenRegion);
	}

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_activateKernel.setArg(argIndex++, vl._reconstructionError);
		_activateKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
		_activateKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_front]);
		_activateKernel.setArg(argIndex++, vl._weights[_back]);
		_activateKernel.setArg(argIndex++, vld._size);
		_activateKernel.setArg(argIndex++, vl._hiddenToVisible);
		_activateKernel.setArg(argIndex++, vld._radius);

		cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		// Swap buffers
		std::swap(_hiddenErrorSummationTemp[_front], _hiddenErrorSummationTemp[_back]);
	}
}

void ComparisonSparseCoder::learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float weightAlpha, float boostAlpha, float activeRatio) {
	// Learn biases
	{
		int argIndex = 0;

		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, weightAlpha);
		_learnHiddenBiasesKernel.setArg(argIndex++, boostAlpha);
		_learnHiddenBiasesKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnHiddenBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
	}
	
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		{
			int argIndex = 0;

			_learnHiddenWeightsKernel.setArg(argIndex++, visibleStates[vli]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._reconstructionError);
			_learnHiddenWeightsKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_learnHiddenWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._weights[_back]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._weights[_front]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vld._size);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnHiddenWeightsKernel.setArg(argIndex++, vld._radius);
			_learnHiddenWeightsKernel.setArg(argIndex++, weightAlpha);

			cs.getQueue().enqueueNDRangeKernel(_learnHiddenWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void ComparisonSparseCoder::learnTrace(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &rewards, float weightAlpha, float weightLambda, float boostAlpha, float activeRatio) {
	// Learn biases
	{
		int argIndex = 0;

		_learnHiddenBiasesTracesKernel.setArg(argIndex++, rewards);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, _hiddenBiases[_back]);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, _hiddenBiases[_front]);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, weightAlpha);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, weightLambda);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, boostAlpha);
		_learnHiddenBiasesTracesKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnHiddenBiasesTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
	}

	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		{
			int argIndex = 0;

			_learnHiddenWeightsTracesKernel.setArg(argIndex++, rewards);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, visibleStates[vli]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._reconstructionError);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._weights[_back]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._weights[_front]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vld._size);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vld._radius);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, weightAlpha);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, weightLambda);

			cs.getQueue().enqueueNDRangeKernel(_learnHiddenWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}
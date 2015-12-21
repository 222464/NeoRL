#include "ComparisonSparseCoder.h"

#include <iostream>

using namespace neo;

void ComparisonSparseCoder::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs,
	cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float initThreshold,
	std::mt19937 &rng)
{
	_visibleLayerDescs = visibleLayerDescs;

	_lateralRadius = lateralRadius;

	_hiddenSize = hiddenSize;

	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	_visibleLayers.resize(_visibleLayerDescs.size());

	cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
	cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

	// Create layers
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		const cl_channel_order weightChannels = vld._useTraces ? CL_RG : CL_R;

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

		cl_int3 weightsSize = cl_int3 { _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, weightChannels, CL_FLOAT);

		randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	randomUniform(_hiddenBiases[_back], cs, randomUniform2DKernel, _hiddenSize, initWeightRange, rng);

	_hiddenActivationSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
	_hiddenErrorSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
	
	// Create kernels
	_forwardErrorKernel = cl::Kernel(program.getProgram(), "cscForwardError");
	_activateKernel = cl::Kernel(program.getProgram(), "cscActivate");
	_activateIgnoreMiddleKernel = cl::Kernel(program.getProgram(), "cscActivateIgnoreMiddle");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "cscSolveHidden");
	_learnHiddenBiasesKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenBiases");
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

		if (vld._ignoreMiddle) {
			int argIndex = 0;

			_activateIgnoreMiddleKernel.setArg(argIndex++, visibleStates[vli]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_front]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vl._weights[_back]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vld._size);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vl._hiddenToVisible);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vld._radius);

			cs.getQueue().enqueueNDRangeKernel(_activateIgnoreMiddleKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}
		else {
			int argIndex = 0;

			_activateKernel.setArg(argIndex++, visibleStates[vli]);
			_activateKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_back]);
			_activateKernel.setArg(argIndex++, _hiddenActivationSummationTemp[_front]);
			_activateKernel.setArg(argIndex++, vl._weights[_back]);
			_activateKernel.setArg(argIndex++, vld._size);
			_activateKernel.setArg(argIndex++, vl._hiddenToVisible);
			_activateKernel.setArg(argIndex++, vld._radius);

			cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

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

		if (vld._ignoreMiddle) {
			int argIndex = 0;

			_activateIgnoreMiddleKernel.setArg(argIndex++, vl._reconstructionError);
			_activateIgnoreMiddleKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_front]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vl._weights[_back]);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vld._size);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vl._hiddenToVisible);
			_activateIgnoreMiddleKernel.setArg(argIndex++, vld._radius);

			cs.getQueue().enqueueNDRangeKernel(_activateIgnoreMiddleKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}
		else {
			int argIndex = 0;

			_activateKernel.setArg(argIndex++, vl._reconstructionError);
			_activateKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_activateKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_front]);
			_activateKernel.setArg(argIndex++, vl._weights[_back]);
			_activateKernel.setArg(argIndex++, vld._size);
			_activateKernel.setArg(argIndex++, vl._hiddenToVisible);
			_activateKernel.setArg(argIndex++, vld._radius);

			cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		// Swap buffers
		std::swap(_hiddenErrorSummationTemp[_front], _hiddenErrorSummationTemp[_back]);
	}
}

void ComparisonSparseCoder::learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float boostAlpha, float activeRatio) {
	// Learn biases
	{
		int argIndex = 0;

		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, boostAlpha);
		_learnHiddenBiasesKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnHiddenBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
	}

	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnHiddenWeightsKernel.setArg(argIndex++, vl._reconstructionError);
		_learnHiddenWeightsKernel.setArg(argIndex++, visibleStates[vli]);
		_learnHiddenWeightsKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
		_learnHiddenWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnHiddenWeightsKernel.setArg(argIndex++, vl._weights[_back]);
		_learnHiddenWeightsKernel.setArg(argIndex++, vl._weights[_front]);
		_learnHiddenWeightsKernel.setArg(argIndex++, vld._size);
		_learnHiddenWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnHiddenWeightsKernel.setArg(argIndex++, vld._radius);
		_learnHiddenWeightsKernel.setArg(argIndex++, vld._weightAlpha);

		cs.getQueue().enqueueNDRangeKernel(_learnHiddenWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void ComparisonSparseCoder::learn(sys::ComputeSystem &cs, const cl::Image2D &rewards, std::vector<cl::Image2D> &visibleStates, float boostAlpha, float activeRatio) {
	// Learn biases
	{
		int argIndex = 0;

		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, _hiddenStates[_back]);
		_learnHiddenBiasesKernel.setArg(argIndex++, boostAlpha);
		_learnHiddenBiasesKernel.setArg(argIndex++, activeRatio);

		cs.getQueue().enqueueNDRangeKernel(_learnHiddenBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
	}
	
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		if (vld._useTraces) {
			int argIndex = 0;

			_learnHiddenWeightsTracesKernel.setArg(argIndex++, rewards);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._reconstructionError);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, visibleStates[vli]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._weights[_back]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._weights[_front]);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vld._size);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vld._radius);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vld._weightAlpha);
			_learnHiddenWeightsTracesKernel.setArg(argIndex++, vld._weightLambda);

			cs.getQueue().enqueueNDRangeKernel(_learnHiddenWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}
		else {
			int argIndex = 0;

			_learnHiddenWeightsKernel.setArg(argIndex++, vl._reconstructionError);
			_learnHiddenWeightsKernel.setArg(argIndex++, visibleStates[vli]);
			_learnHiddenWeightsKernel.setArg(argIndex++, _hiddenErrorSummationTemp[_back]);
			_learnHiddenWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._weights[_back]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._weights[_front]);
			_learnHiddenWeightsKernel.setArg(argIndex++, vld._size);
			_learnHiddenWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
			_learnHiddenWeightsKernel.setArg(argIndex++, vld._radius);
			_learnHiddenWeightsKernel.setArg(argIndex++, vld._weightAlpha);

			cs.getQueue().enqueueNDRangeKernel(_learnHiddenWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void ComparisonSparseCoder::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	os << _hiddenSize.x << " " << _hiddenSize.y << " " << _lateralRadius << std::endl;

	{
		std::vector<cl_float> hiddenStates(_hiddenSize.x * _hiddenSize.y);

		cs.getQueue().enqueueReadImage(_hiddenStates[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenStates.data());

		for (int si = 0; si < hiddenStates.size(); si++)
			os << hiddenStates[si] << " ";

		os << std::endl;
	}

	{
		std::vector<cl_float> hiddenBiases(_hiddenSize.x * _hiddenSize.y);

		cs.getQueue().enqueueReadImage(_hiddenBiases[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenBiases.data());

		for (int bi = 0; bi < hiddenBiases.size(); bi++)
			os << hiddenBiases[bi] << " ";

		os << std::endl;
	}

	// Layer information
	os << _visibleLayers.size() << std::endl;

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		const VisibleLayer &vl = _visibleLayers[vli];
		const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		// Desc
		os << vld._size.x << " " << vld._size.y << " " << vld._radius << " " << vld._weightAlpha << " " << vld._weightLambda << " " << vld._ignoreMiddle << " " << vld._useTraces << std::endl;

		// Layer
		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3 { _hiddenSize.x, _hiddenSize.y, numWeights };

		int totalNumWeights = weightsSize.x * weightsSize.y * weightsSize.z;

		if (vld._useTraces) {
			std::vector<cl_float2> weights(totalNumWeights);

			cs.getQueue().enqueueReadImage(vl._weights[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(weightsSize.x), static_cast<cl::size_type>(weightsSize.y), static_cast<cl::size_type>(weightsSize.z) }, 0, 0, weights.data());
		
			for (int wi = 0; wi < weights.size(); wi++)
				os << weights[wi].x << " " << weights[wi].y << " ";
		}
		else {
			std::vector<cl_float> weights(totalNumWeights);

			cs.getQueue().enqueueReadImage(vl._weights[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(weightsSize.x), static_cast<cl::size_type>(weightsSize.y), static_cast<cl::size_type>(weightsSize.z) }, 0, 0, weights.data());

			for (int wi = 0; wi < weights.size(); wi++)
				os << weights[wi] << " ";
		}

		os << std::endl;

		os << vl._hiddenToVisible.x << " " << vl._hiddenToVisible.y << " " << vl._visibleToHidden.x << " " << vl._visibleToHidden.y << " " << vl._reverseRadii.x << " " << vl._reverseRadii.y << std::endl;
	}
}
void ComparisonSparseCoder::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
	is >> _hiddenSize.x >> _hiddenSize.y >> _lateralRadius;

	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenActivationSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
	_hiddenErrorSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	{
		std::vector<cl_float> hiddenStates(_hiddenSize.x * _hiddenSize.y);

		for (int si = 0; si < hiddenStates.size(); si++)
			is >> hiddenStates[si];

		cs.getQueue().enqueueWriteImage(_hiddenStates[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenStates.data());

	}

	{
		std::vector<cl_float> hiddenBiases(_hiddenSize.x * _hiddenSize.y);

		for (int bi = 0; bi < hiddenBiases.size(); bi++)
			is >> hiddenBiases[bi];

		cs.getQueue().enqueueWriteImage(_hiddenBiases[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenBiases.data());
	}

	// Layer information
	int numLayers;

	is >> numLayers;

	_visibleLayerDescs.resize(numLayers);
	_visibleLayers.resize(numLayers);

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		// Desc
		is >> vld._size.x >> vld._size.y >> vld._radius >> vld._weightAlpha >> vld._weightLambda >> vld._ignoreMiddle >> vld._useTraces;

		// Layer
		vl._reconstructionError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3 { _hiddenSize.x, _hiddenSize.y, numWeights };

		int totalNumWeights = weightsSize.x * weightsSize.y * weightsSize.z;

		if (vld._useTraces) {
			vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);
			
			std::vector<cl_float2> weights(totalNumWeights);
	
			for (int wi = 0; wi < weights.size(); wi++)
				is >> weights[wi].x >> weights[wi].y;

			cs.getQueue().enqueueWriteImage(vl._weights[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(weightsSize.x), static_cast<cl::size_type>(weightsSize.y), static_cast<cl::size_type>(weightsSize.z) }, 0, 0, weights.data());
		}
		else {
			vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

			std::vector<cl_float> weights(totalNumWeights);

			for (int wi = 0; wi < weights.size(); wi++)
				is >> weights[wi];

			cs.getQueue().enqueueWriteImage(vl._weights[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(weightsSize.x), static_cast<cl::size_type>(weightsSize.y), static_cast<cl::size_type>(weightsSize.z) }, 0, 0, weights.data());
		}

		is >> vl._hiddenToVisible.x >> vl._hiddenToVisible.y >> vl._visibleToHidden.x >> vl._visibleToHidden.y >> vl._reverseRadii.x >> vl._reverseRadii.y;
	}

	// Create kernels
	_forwardErrorKernel = cl::Kernel(program.getProgram(), "cscForwardError");
	_activateKernel = cl::Kernel(program.getProgram(), "cscActivate");
	_activateIgnoreMiddleKernel = cl::Kernel(program.getProgram(), "cscActivateIgnoreMiddle");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "cscSolveHidden");
	_learnHiddenBiasesKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenBiases");
	_learnHiddenWeightsKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenWeights");
	_learnHiddenWeightsTracesKernel = cl::Kernel(program.getProgram(), "cscLearnHiddenWeightsTraces");
}

void ComparisonSparseCoder::clearMemory(sys::ComputeSystem &cs) {
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };
	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };

	cl::array<cl::size_type, 3> layerRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, layerRegion);
}
#include "Predictor.h"

using namespace neo;

void Predictor::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

	_visibleLayerDescs = visibleLayerDescs;

	_hiddenSize = hiddenSize;

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

		vl._reverseRadii = cl_int2{ static_cast<int>(std::ceil(vl._visibleToHidden.x * vld._radius)), static_cast<int>(std::ceil(vl._visibleToHidden.y * vld._radius)) };

		vl._errors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		cs.getQueue().enqueueFillImage(vl._errors, zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

		vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

		randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
	}

	// Hidden state data
	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
	cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);

	// Create kernels
	_activateKernel = cl::Kernel(program.getProgram(), "predActivate");
	_solveHiddenThresholdKernel = cl::Kernel(program.getProgram(), "predSolveHiddenThreshold");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "predSolveHidden");
	_errorPropagateKernel = cl::Kernel(program.getProgram(), "predErrorPropagate");
	_learnWeightsKernel = cl::Kernel(program.getProgram(), "predLearnWeights");
	_learnWeightsTracesKernel = cl::Kernel(program.getProgram(), "predLearnWeightsTraces");
}

void Predictor::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, bool threshold) {
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

	if (threshold) {
		int argIndex = 0;

		_solveHiddenThresholdKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_solveHiddenThresholdKernel.setArg(argIndex++, _hiddenStates[_back]);
		_solveHiddenThresholdKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenThresholdKernel.setArg(argIndex++, _hiddenActivations[_back]);
		_solveHiddenThresholdKernel.setArg(argIndex++, _hiddenActivations[_front]);

		cs.getQueue().enqueueNDRangeKernel(_solveHiddenThresholdKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}
	else {
		int argIndex = 0;

		_solveHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenActivations[_back]);
		_solveHiddenKernel.setArg(argIndex++, _hiddenActivations[_front]);

		cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
	}

	// Swap hidden state buffers
	std::swap(_hiddenStates[_front], _hiddenStates[_back]);
	std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
}

void Predictor::propagateError(sys::ComputeSystem &cs, const cl::Image2D &targets) {
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_errorPropagateKernel.setArg(argIndex++, targets);
		_errorPropagateKernel.setArg(argIndex++, _hiddenStates[_front]);
		_errorPropagateKernel.setArg(argIndex++, vl._errors);
		_errorPropagateKernel.setArg(argIndex++, vl._weights[_back]);
		_errorPropagateKernel.setArg(argIndex++, vld._size);
		_errorPropagateKernel.setArg(argIndex++, _hiddenSize);
		_errorPropagateKernel.setArg(argIndex++, vl._visibleToHidden);
		_errorPropagateKernel.setArg(argIndex++, vl._hiddenToVisible);
		_errorPropagateKernel.setArg(argIndex++, vld._radius);
		_errorPropagateKernel.setArg(argIndex++, vl._reverseRadii);

		cs.getQueue().enqueueNDRangeKernel(_errorPropagateKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
	}
}

void Predictor::learn(sys::ComputeSystem &cs, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, float weightAlpha) {
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsKernel.setArg(argIndex++, visibleStatesPrev[vli]);
		_learnWeightsKernel.setArg(argIndex++, targets);
		_learnWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
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

void Predictor::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	os << _hiddenSize.x << " " << _hiddenSize.y << std::endl;

	{
		std::vector<cl_float> hiddenStates(_hiddenSize.x * _hiddenSize.y);

		cs.getQueue().enqueueReadImage(_hiddenStates[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenStates.data());

		for (int si = 0; si < hiddenStates.size(); si++)
			os << hiddenStates[si] << " ";

		os << std::endl;
	}

	{
		std::vector<cl_float> hiddenActivations(_hiddenSize.x * _hiddenSize.y);

		cs.getQueue().enqueueReadImage(_hiddenActivations[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenActivations.data());

		for (int bi = 0; bi < hiddenActivations.size(); bi++)
			os << hiddenActivations[bi] << " ";

		os << std::endl;
	}

	// Layer information
	os << _visibleLayers.size() << std::endl;

	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		const VisibleLayer &vl = _visibleLayers[vli];
		const VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		// Desc
		os << vld._size.x << " " << vld._size.y << " " << vld._radius << std::endl;

		// Layer
		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3{ _hiddenSize.x, _hiddenSize.y, numWeights };

		int totalNumWeights = weightsSize.x * weightsSize.y * weightsSize.z;

		{
			std::vector<cl_float> weights(totalNumWeights);

			cs.getQueue().enqueueReadImage(vl._weights[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(weightsSize.x), static_cast<cl::size_type>(weightsSize.y), static_cast<cl::size_type>(weightsSize.z) }, 0, 0, weights.data());

			for (int wi = 0; wi < weights.size(); wi++)
				os << weights[wi] << " ";
		}

		os << std::endl;

		os << vl._hiddenToVisible.x << " " << vl._hiddenToVisible.y << " " << vl._visibleToHidden.x << " " << vl._visibleToHidden.y << " " << vl._reverseRadii.x << " " << vl._reverseRadii.y << std::endl;
	}
}
void Predictor::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
	is >> _hiddenSize.x >> _hiddenSize.y;

	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	{
		std::vector<cl_float> hiddenStates(_hiddenSize.x * _hiddenSize.y);

		for (int si = 0; si < hiddenStates.size(); si++)
			is >> hiddenStates[si];

		cs.getQueue().enqueueWriteImage(_hiddenStates[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenStates.data());

	}

	{
		std::vector<cl_float> hiddenActivations(_hiddenSize.x * _hiddenSize.y);

		for (int bi = 0; bi < hiddenActivations.size(); bi++)
			is >> hiddenActivations[bi];

		cs.getQueue().enqueueWriteImage(_hiddenActivations[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenActivations.data());
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
		is >> vld._size.x >> vld._size.y >> vld._radius;

		// Layer
		vl._errors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);

		int weightDiam = vld._radius * 2 + 1;

		int numWeights = weightDiam * weightDiam;

		cl_int3 weightsSize = cl_int3{ _hiddenSize.x, _hiddenSize.y, numWeights };

		int totalNumWeights = weightsSize.x * weightsSize.y * weightsSize.z;

		{
			vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

			std::vector<cl_float> weights(totalNumWeights);

			for (int wi = 0; wi < weights.size(); wi++)
				is >> weights[wi];

			cs.getQueue().enqueueWriteImage(vl._weights[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(weightsSize.x), static_cast<cl::size_type>(weightsSize.y), static_cast<cl::size_type>(weightsSize.z) }, 0, 0, weights.data());
		}

		is >> vl._hiddenToVisible.x >> vl._hiddenToVisible.y >> vl._visibleToHidden.x >> vl._visibleToHidden.y >> vl._reverseRadii.x >> vl._reverseRadii.y;
	}

	// Create kernels
	_activateKernel = cl::Kernel(program.getProgram(), "predActivate");
	_solveHiddenThresholdKernel = cl::Kernel(program.getProgram(), "predSolveHiddenThreshold");
	_solveHiddenKernel = cl::Kernel(program.getProgram(), "predSolveHidden");
	_errorPropagateKernel = cl::Kernel(program.getProgram(), "predErrorPropagate");
	_learnWeightsKernel = cl::Kernel(program.getProgram(), "predLearnWeights");
	_learnWeightsTracesKernel = cl::Kernel(program.getProgram(), "predLearnWeightsTraces");
}
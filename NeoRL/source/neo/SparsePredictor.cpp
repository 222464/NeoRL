#include "SparsePredictor.h"

using namespace neo;

void SparsePredictor::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange,
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
	_solveVisibleKernel = cl::Kernel(program.getProgram(), "spSolveVisible");
	_predictionErrorKernel = cl::Kernel(program.getProgram(), "spPredictionError");
	_errorPropagationKernel = cl::Kernel(program.getProgram(), "spErrorPropagation");
	_learnEncoderWeightsKernel = cl::Kernel(program.getProgram(), "spLearnEncoderWeights");
	_learnDecoderWeightsKernel = cl::Kernel(program.getProgram(), "spLearnDecoderWeights");
}

void SparsePredictor::activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const std::vector<cl::Image2D> &feedBackStates, float activeRatio) {
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
		_decodeKernel.setArg(argIndex++, vl._visibleToHidden);
		_decodeKernel.setArg(argIndex++, vld._predDecodeRadius);

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

void SparsePredictor::learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const std::vector<cl::Image2D> &visibleStatesPrev,
	const std::vector<cl::Image2D> &feedBackStatesPrev, const std::vector<cl::Image2D> &addidionalErrors, float weightAlpha, float weightLambda)
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

			cs.getQueue().enqueueNDRangeKernel(_predictionErrorKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
		}

		// Propagate the error
		{
			cl_int2 reversePredDecodeRadii = { static_cast<int>(std::ceil(vl._visibleToHidden.x * (vld._predDecodeRadius + 0.5f))), static_cast<int>(std::ceil(vl._visibleToHidden.y * (vld._predDecodeRadius + 0.5f))) };
			
			int argIndex = 0;

			/*read_only image2d_t targets, read_only image2d_t hiddenStatesPrev,
	write_only image2d_t errors, read_only image3d_t weights,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii*/
			_errorPropagationKernel.setArg(argIndex++, )

			cs.getQueue().enqueueNDRangeKernel(_errorPropagationKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
		}
	}
	
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

void SparsePredictor::learn(sys::ComputeSystem &cs, float tdError, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, float weightAlpha, float weightLambda) {
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsTracesKernel.setArg(argIndex++, visibleStatesPrev[vli]);
		_learnWeightsTracesKernel.setArg(argIndex++, targets);
		_learnWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_front]);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._weights[_back]);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._weights[_front]);
		_learnWeightsTracesKernel.setArg(argIndex++, vld._size);
		_learnWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnWeightsTracesKernel.setArg(argIndex++, vld._radius);
		_learnWeightsTracesKernel.setArg(argIndex++, weightAlpha);
		_learnWeightsTracesKernel.setArg(argIndex++, weightLambda);
		_learnWeightsTracesKernel.setArg(argIndex++, tdError);

		cs.getQueue().enqueueNDRangeKernel(_learnWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void SparsePredictor::learnQ(sys::ComputeSystem &cs, float tdError, std::vector<cl::Image2D> &visibleStatesPrev, float weightAlpha, float weightLambda) {
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnQWeightsTracesKernel.setArg(argIndex++, visibleStatesPrev[vli]);
		_learnQWeightsTracesKernel.setArg(argIndex++, _hiddenStates[_front]);
		_learnQWeightsTracesKernel.setArg(argIndex++, vl._weights[_back]);
		_learnQWeightsTracesKernel.setArg(argIndex++, vl._weights[_front]);
		_learnQWeightsTracesKernel.setArg(argIndex++, vld._size);
		_learnQWeightsTracesKernel.setArg(argIndex++, vl._hiddenToVisible);
		_learnQWeightsTracesKernel.setArg(argIndex++, vld._radius);
		_learnQWeightsTracesKernel.setArg(argIndex++, weightAlpha);
		_learnQWeightsTracesKernel.setArg(argIndex++, weightLambda);
		_learnQWeightsTracesKernel.setArg(argIndex++, tdError);

		cs.getQueue().enqueueNDRangeKernel(_learnQWeightsTracesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

		std::swap(vl._weights[_front], vl._weights[_back]);
	}
}

void SparsePredictor::learnCurrent(sys::ComputeSystem &cs, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStates, float weightAlpha) {
	// Learn weights
	for (int vli = 0; vli < _visibleLayers.size(); vli++) {
		VisibleLayer &vl = _visibleLayers[vli];
		VisibleLayerDesc &vld = _visibleLayerDescs[vli];

		int argIndex = 0;

		_learnWeightsKernel.setArg(argIndex++, visibleStates[vli]);
		_learnWeightsKernel.setArg(argIndex++, targets);
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
}

void SparsePredictor::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	abort(); // Not yet working

	os << _hiddenSize.x << " " << _hiddenSize.y << std::endl;

	{
		std::vector<cl_float> hiddenStates(_hiddenSize.x * _hiddenSize.y);

		cs.getQueue().enqueueReadImage(_hiddenStates[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenStates.data());

		for (int si = 0; si < hiddenStates.size(); si++)
			os << hiddenStates[si] << " ";

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
void SparsePredictor::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
	abort(); // Not yet working

	is >> _hiddenSize.x >> _hiddenSize.y;

	_hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	_hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

	{
		std::vector<cl_float> hiddenStates(_hiddenSize.x * _hiddenSize.y);

		for (int si = 0; si < hiddenStates.size(); si++)
			is >> hiddenStates[si];

		cs.getQueue().enqueueWriteImage(_hiddenStates[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 }, 0, 0, hiddenStates.data());

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
	//_solveHiddenKernel = cl::Kernel(program.getProgram(), "predSolveHidden");
	_learnWeightsKernel = cl::Kernel(program.getProgram(), "predLearnWeights");
}
#include "BIDInet.h"

#include <iostream>

using namespace bidi;

void BIDInet::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program, int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, std::mt19937 &generator) {
	_inputWidth = inputWidth;
	_inputHeight = inputHeight;

	int numInputs = inputWidth * inputHeight;

	// Inputs
	_inputs = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputWidth, _inputHeight);

	_inputsTemp.clear();
	_inputsTemp.assign(numInputs, 0.0f);

	_outputsTemp.clear();
	_outputsTemp.assign(numInputs, 0.0f);

	// Q connections
	std::uniform_real_distribution<float> initWeightDist(initMinWeight, initMaxWeight);

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	cl::Kernel initializeConnectionsKernel = cl::Kernel(program.getProgram(), "initializeConnections");

	std::uniform_int_distribution<int> seedDist(0, 10000);

	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	for (int l = 0; l < _layers.size(); l++) {
		Layer &layer = _layers[l];

		LayerDesc &layerDesc = _layerDescs[l];

		int ffDiam = layerDesc._ffRadius * 2 + 1;
		int lDiam = layerDesc._lRadius * 2 + 1;
		int recDiam = layerDesc._recRadius * 2 + 1;
		int fbDiam = layerDesc._fbRadius * 2 + 1;
		int predDiam = layerDesc._predRadius * 2 + 1;

		// + 1 for biases (if applicable)
		int ffSize = ffDiam * ffDiam + 1;
		int lSize = lDiam * lDiam;
		int recSize = recDiam * recDiam;
		int fbSize = fbDiam * fbDiam + 1;
		int predSize = predDiam * predDiam;

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { layerDesc._width, layerDesc._height, 1 };
		cl::array<cl::size_type, 3> prevLayerRegion = { prevLayerWidth, prevLayerHeight, 1 };

		cl_uint4 zeroColor = { 0, 0, 0, 0 };
		
		// Activations
		layer._ffActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
		layer._fbActivations = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth,prevLayerHeight);
		layer._fbActivationsExploratory = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth, prevLayerHeight);

		// Reconstruction
		layer._ffReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth, prevLayerHeight);

		layer._recReconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

		// States
		layer._ffStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);
		layer._ffStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height);

		cs.getQueue().enqueueFillImage(layer._ffStatesPrev, zeroColor, zeroOrigin, layerRegion);

		layer._fbStates = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth, prevLayerHeight);
		layer._fbStatesExploratory = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth, prevLayerHeight);
		layer._fbStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth, prevLayerHeight);
		layer._fbStatesExploratoryPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), prevLayerWidth, prevLayerHeight);

		cs.getQueue().enqueueFillImage(layer._fbStatesPrev, zeroColor, zeroOrigin, prevLayerRegion);
		cs.getQueue().enqueueFillImage(layer._fbStatesExploratoryPrev, zeroColor, zeroOrigin, prevLayerRegion);

		// Connections
		{
			layer._ffConnections = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, ffSize);
			layer._ffConnectionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, ffSize);

			int argIndex = 0;

			cl_uint2 seed = { seedDist(generator), seedDist(generator) };

			initializeConnectionsKernel.setArg(argIndex++, layer._ffConnectionsPrev);
			initializeConnectionsKernel.setArg(argIndex++, ffSize);
			initializeConnectionsKernel.setArg(argIndex++, seed);
			initializeConnectionsKernel.setArg(argIndex++, initMinWeight);
			initializeConnectionsKernel.setArg(argIndex++, initMaxWeight);
		
			cs.getQueue().enqueueNDRangeKernel(initializeConnectionsKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		{
			layer._recConnections = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, recSize);
			layer._recConnectionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), layerDesc._width, layerDesc._height, recSize);

			int argIndex = 0;

			cl_uint2 seed = { seedDist(generator), seedDist(generator) };

			initializeConnectionsKernel.setArg(argIndex++, layer._recConnectionsPrev);
			initializeConnectionsKernel.setArg(argIndex++, recSize);
			initializeConnectionsKernel.setArg(argIndex++, seed);
			initializeConnectionsKernel.setArg(argIndex++, initMinWeight);
			initializeConnectionsKernel.setArg(argIndex++, initMaxWeight);

			cs.getQueue().enqueueNDRangeKernel(initializeConnectionsKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		{
			layer._fbConnections = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevLayerWidth, prevLayerHeight, fbSize);
			layer._fbConnectionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevLayerWidth, prevLayerHeight, fbSize);

			int argIndex = 0;

			cl_uint2 seed = { seedDist(generator), seedDist(generator) };

			initializeConnectionsKernel.setArg(argIndex++, layer._fbConnectionsPrev);
			initializeConnectionsKernel.setArg(argIndex++, fbSize);
			initializeConnectionsKernel.setArg(argIndex++, seed);
			initializeConnectionsKernel.setArg(argIndex++, initMinWeight);
			initializeConnectionsKernel.setArg(argIndex++, initMaxWeight);

			cs.getQueue().enqueueNDRangeKernel(initializeConnectionsKernel, cl::NullRange, cl::NDRange(prevLayerWidth, prevLayerHeight));
		}

		{
			layer._predConnections = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevLayerWidth, prevLayerHeight, predSize);
			layer._predConnectionsPrev = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), prevLayerWidth, prevLayerHeight, predSize);

			int argIndex = 0;

			cl_uint2 seed = { seedDist(generator), seedDist(generator) };

			initializeConnectionsKernel.setArg(argIndex++, layer._predConnectionsPrev);
			initializeConnectionsKernel.setArg(argIndex++, predSize);
			initializeConnectionsKernel.setArg(argIndex++, seed);
			initializeConnectionsKernel.setArg(argIndex++, initMinWeight);
			initializeConnectionsKernel.setArg(argIndex++, initMaxWeight);

			cs.getQueue().enqueueNDRangeKernel(initializeConnectionsKernel, cl::NullRange, cl::NDRange(prevLayerWidth, prevLayerHeight));
		}

		// Update prevs
		prevLayerWidth = layerDesc._width;
		prevLayerHeight = layerDesc._height;
	}

	// Get kernels
	_ffActivateKernel = cl::Kernel(program.getProgram(), "ffActivate");
	_ffInhibitKernel = cl::Kernel(program.getProgram(), "ffInhibit");
	_fbActivateKernel = cl::Kernel(program.getProgram(), "fbActivate");
	_fbActivateFirstKernel = cl::Kernel(program.getProgram(), "fbActivateFirst");
	_ffReconstructKernel = cl::Kernel(program.getProgram(), "ffReconstruct");
	_recReconstructKernel = cl::Kernel(program.getProgram(), "recReconstruct");
	_ffConnectionUpdateKernel = cl::Kernel(program.getProgram(), "ffConnectionUpdate");
	_recConnectionUpdateKernel = cl::Kernel(program.getProgram(), "recConnectionUpdate");
	_fbConnectionUpdateKernel = cl::Kernel(program.getProgram(), "fbConnectionUpdate");
	_predConnectionUpdateKernel = cl::Kernel(program.getProgram(), "predConnectionUpdate");
}

void BIDInet::simStep(sys::ComputeSystem &cs, float reward, float breakChance, std::mt19937 &generator) {
	float rlError = reward;
	
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> inputRegion = { _inputWidth, _inputHeight, 1 };

	for (int a = 0; a < _actionIndices.size(); a++)
		_inputsTemp[_actionIndices[a]] = _outputsTemp[_actionIndices[a]];
	
	cs.getQueue().enqueueWriteImage(_inputs, CL_TRUE, zeroOrigin, inputRegion, 0, 0, _inputsTemp.data());

	// Feed forward
	int prevLayerWidth = _inputWidth;
	int prevLayerHeight = _inputHeight;

	cl::Image2D* pPrevLayer = &_inputs;

	for (int l = 0; l < _layers.size(); l++) {
		Layer &layer = _layers[l];

		LayerDesc &layerDesc = _layerDescs[l];

		cl_int2 layerSize = { layerDesc._width, layerDesc._height };
		cl_int2 inputsSize = { prevLayerWidth, prevLayerHeight };
		
		// Activate
		{
			cl_float2 layerToInputsScalar = { static_cast<float>(inputsSize.x) / static_cast<float>(layerSize.x),
				static_cast<float>(inputsSize.y) / static_cast<float>(layerSize.y) };

			int argIndex = 0;

			_ffActivateKernel.setArg(argIndex++, *pPrevLayer);
			_ffActivateKernel.setArg(argIndex++, layer._ffStatesPrev);
			_ffActivateKernel.setArg(argIndex++, layer._ffConnectionsPrev);
			_ffActivateKernel.setArg(argIndex++, layer._recConnectionsPrev);
			_ffActivateKernel.setArg(argIndex++, layer._ffActivations);
			_ffActivateKernel.setArg(argIndex++, layerSize);
			_ffActivateKernel.setArg(argIndex++, inputsSize);
			_ffActivateKernel.setArg(argIndex++, layerDesc._ffRadius);
			_ffActivateKernel.setArg(argIndex++, layerDesc._recRadius);
			_ffActivateKernel.setArg(argIndex++, layerToInputsScalar);

			cs.getQueue().enqueueNDRangeKernel(_ffActivateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		// Inhibit
		{
			int lDiam = layerDesc._lRadius * 2 + 1;

			int lSize = lDiam * lDiam;

			float numActive = _layerDescs[l]._sparsity * lSize;

			int argIndex = 0;

			_ffInhibitKernel.setArg(argIndex++, layer._ffActivations);
			_ffInhibitKernel.setArg(argIndex++, layer._ffStates);
			_ffInhibitKernel.setArg(argIndex++, layerSize);
			_ffInhibitKernel.setArg(argIndex++, layerDesc._lRadius);
			_ffInhibitKernel.setArg(argIndex++, numActive);
			_ffInhibitKernel.setArg(argIndex++, layerDesc._sparsity);

			cs.getQueue().enqueueNDRangeKernel(_ffInhibitKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		// Update prevs
		prevLayerWidth = layerDesc._width;
		prevLayerHeight = layerDesc._height;

		pPrevLayer = &layer._ffStates;
	}

	std::uniform_int_distribution<int> seedDist(0, 10000);

	// Feed back
	for (int l = _layers.size() - 1; l >= 0; l--) {
		Layer &layer = _layers[l];

		LayerDesc &layerDesc = _layerDescs[l];

		cl_int2 layerSize = { layerDesc._width, layerDesc._height };

		cl_uint2 seed = { seedDist(generator), seedDist(generator) };

		cl_int2 nextSize;
		
		if (l == 0) {
			nextSize.x = _inputWidth;
			nextSize.y = _inputHeight;

			cl_float2 layerToInputsScalar = { static_cast<float>(layerSize.x) / static_cast<float>(nextSize.x),
				static_cast<float>(layerSize.y) / static_cast<float>(nextSize.y) };

			int argIndex = 0;

			if (l == _layers.size() - 1)
				_fbActivateFirstKernel.setArg(argIndex++, layer._ffStates);
			else
				_fbActivateFirstKernel.setArg(argIndex++, _layers[l + 1]._fbStatesExploratory);

			_fbActivateFirstKernel.setArg(argIndex++, layer._fbConnectionsPrev);
			_fbActivateFirstKernel.setArg(argIndex++, layer._predConnectionsPrev);
			_fbActivateFirstKernel.setArg(argIndex++, _layers[l]._fbActivations);
			_fbActivateFirstKernel.setArg(argIndex++, _layers[l]._fbActivationsExploratory);
			_fbActivateFirstKernel.setArg(argIndex++, layerSize);
			_fbActivateFirstKernel.setArg(argIndex++, nextSize);
			_fbActivateFirstKernel.setArg(argIndex++, layerDesc._fbRadius);
			_fbActivateFirstKernel.setArg(argIndex++, layerDesc._predRadius);
			_fbActivateFirstKernel.setArg(argIndex++, layerToInputsScalar);
			_fbActivateFirstKernel.setArg(argIndex++, breakChance);
			_fbActivateFirstKernel.setArg(argIndex++, seed);

			cs.getQueue().enqueueNDRangeKernel(_fbActivateFirstKernel, cl::NullRange, cl::NDRange(nextSize.x, nextSize.y));
		}
		else {
			nextSize.x = _layerDescs[l - 1]._width;
			nextSize.y = _layerDescs[l - 1]._height;
		
			cl_float2 layerToInputsScalar = { static_cast<float>(layerSize.x) / static_cast<float>(nextSize.x),
				static_cast<float>(layerSize.y) / static_cast<float>(nextSize.y) };

			int argIndex = 0;

			if (l == _layers.size() - 1)
				_fbActivateKernel.setArg(argIndex++, layer._ffStates);
			else
				_fbActivateKernel.setArg(argIndex++, _layers[l + 1]._fbStatesExploratory);

			_fbActivateKernel.setArg(argIndex++, _layers[l - 1]._ffStates);
			_fbActivateKernel.setArg(argIndex++, layer._fbConnectionsPrev);
			_fbActivateKernel.setArg(argIndex++, layer._predConnectionsPrev);
			_fbActivateKernel.setArg(argIndex++, _layers[l]._fbActivations);
			_fbActivateKernel.setArg(argIndex++, _layers[l]._fbActivationsExploratory);
			_fbActivateKernel.setArg(argIndex++, layerSize);
			_fbActivateKernel.setArg(argIndex++, nextSize);
			_fbActivateKernel.setArg(argIndex++, layerDesc._fbRadius);
			_fbActivateKernel.setArg(argIndex++, layerDesc._predRadius);
			_fbActivateKernel.setArg(argIndex++, layerToInputsScalar);
			_fbActivateKernel.setArg(argIndex++, breakChance);
			_fbActivateKernel.setArg(argIndex++, seed);

			cs.getQueue().enqueueNDRangeKernel(_fbActivateKernel, cl::NullRange, cl::NDRange(nextSize.x, nextSize.y));
		}

		// Inhibit
		if (l == 0) {
			cl::array<cl::size_type, 3> nextRegion = { nextSize.x, nextSize.y, 1 };

			cs.getQueue().enqueueCopyImage(layer._fbActivations, layer._fbStates, zeroOrigin, zeroOrigin, nextRegion);
			cs.getQueue().enqueueCopyImage(layer._fbActivationsExploratory, layer._fbStatesExploratory, zeroOrigin, zeroOrigin, nextRegion);
		}
		else {
			int lDiam = layerDesc._lRadius * 2 + 1;

			int lSize = lDiam * lDiam;

			float numActive = _layerDescs[l]._sparsity * lSize;

			{
				int argIndex = 0;

				_ffInhibitKernel.setArg(argIndex++, layer._fbActivations);
				_ffInhibitKernel.setArg(argIndex++, layer._fbStates);
				_ffInhibitKernel.setArg(argIndex++, layerSize);
				_ffInhibitKernel.setArg(argIndex++, layerDesc._lRadius);
				_ffInhibitKernel.setArg(argIndex++, numActive);
				_ffInhibitKernel.setArg(argIndex++, layerDesc._sparsity);

				cs.getQueue().enqueueNDRangeKernel(_ffInhibitKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
			}

			{
				int argIndex = 0;

				_ffInhibitKernel.setArg(argIndex++, layer._fbActivationsExploratory);
				_ffInhibitKernel.setArg(argIndex++, layer._fbStatesExploratory);
				_ffInhibitKernel.setArg(argIndex++, layerSize);
				_ffInhibitKernel.setArg(argIndex++, layerDesc._lRadius);
				_ffInhibitKernel.setArg(argIndex++, numActive);
				_ffInhibitKernel.setArg(argIndex++, layerDesc._sparsity);

				cs.getQueue().enqueueNDRangeKernel(_ffInhibitKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
			}
		}
	}

	std::vector<float> outputsTempPrev = _outputsTemp;

	// Get outputs
	cs.getQueue().enqueueReadImage(_layers.front()._fbStatesExploratory, CL_TRUE, zeroOrigin, inputRegion, 0, 0, _outputsTemp.data());

	prevLayerWidth = _inputWidth;
	prevLayerHeight = _inputHeight;

	pPrevLayer = &_inputs;

	for (int l = 0; l < _layers.size(); l++) {
		Layer &layer = _layers[l];

		LayerDesc &layerDesc = _layerDescs[l];

		cl_int2 layerSize = { layerDesc._width, layerDesc._height };
		cl_int2 inputsSize = { prevLayerWidth, prevLayerHeight };

		cl_float2 layerToInputsScalar = { static_cast<float>(inputsSize.x) / static_cast<float>(layerSize.x),
			static_cast<float>(inputsSize.y) / static_cast<float>(layerSize.y) };

		cl_float2 inputsToLayerScalar = { static_cast<float>(layerSize.x) / static_cast<float>(inputsSize.x),
			static_cast<float>(layerSize.y) / static_cast<float>(inputsSize.y) };

		// FF reconstruct
		{
			cl_int2 ffReverseRadius = { std::ceil(layerDesc._ffRadius * inputsToLayerScalar.x + 0.5f),
				std::ceil(layerDesc._ffRadius * inputsToLayerScalar.y + 0.5f) };

			int argIndex = 0;

			_ffReconstructKernel.setArg(argIndex++, layer._ffStates);
			_ffReconstructKernel.setArg(argIndex++, layer._ffConnectionsPrev);
			_ffReconstructKernel.setArg(argIndex++, layer._ffReconstruction); 
			_ffReconstructKernel.setArg(argIndex++, layerDesc._ffRadius);
			_ffReconstructKernel.setArg(argIndex++, ffReverseRadius);
			_ffReconstructKernel.setArg(argIndex++, inputsSize);
			_ffReconstructKernel.setArg(argIndex++, layerSize);
			_ffReconstructKernel.setArg(argIndex++, layerToInputsScalar);
			_ffReconstructKernel.setArg(argIndex++, inputsToLayerScalar);

			cs.getQueue().enqueueNDRangeKernel(_ffReconstructKernel, cl::NullRange, cl::NDRange(prevLayerWidth, prevLayerHeight));
		}

		// Rec reconstruct
		{
			int argIndex = 0;

			_recReconstructKernel.setArg(argIndex++, layer._ffStates);
			_recReconstructKernel.setArg(argIndex++, layer._recConnectionsPrev);
			_recReconstructKernel.setArg(argIndex++, layer._recReconstruction);
			_recReconstructKernel.setArg(argIndex++, layerDesc._recRadius);
			_recReconstructKernel.setArg(argIndex++, layerSize);

			cs.getQueue().enqueueNDRangeKernel(_recReconstructKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		// FF
		{
			int argIndex = 0;

			_ffConnectionUpdateKernel.setArg(argIndex++, *pPrevLayer);
			_ffConnectionUpdateKernel.setArg(argIndex++, layer._ffConnectionsPrev);
			_ffConnectionUpdateKernel.setArg(argIndex++, layer._ffConnections);
			_ffConnectionUpdateKernel.setArg(argIndex++, layer._ffReconstruction);
			_ffConnectionUpdateKernel.setArg(argIndex++, layer._ffStates);
			_ffConnectionUpdateKernel.setArg(argIndex++, layerSize);
			_ffConnectionUpdateKernel.setArg(argIndex++, inputsSize);
			_ffConnectionUpdateKernel.setArg(argIndex++, layerDesc._ffRadius);
			_ffConnectionUpdateKernel.setArg(argIndex++, layerToInputsScalar);
			_ffConnectionUpdateKernel.setArg(argIndex++, layerDesc._ffAlpha);
			_ffConnectionUpdateKernel.setArg(argIndex++, layerDesc._ffGamma);
			_ffConnectionUpdateKernel.setArg(argIndex++, layerDesc._sparsity);

			cs.getQueue().enqueueNDRangeKernel(_ffConnectionUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		// Rec
		{
			int argIndex = 0;

			_recConnectionUpdateKernel.setArg(argIndex++, layer._ffStatesPrev);
			_recConnectionUpdateKernel.setArg(argIndex++, layer._recConnectionsPrev);
			_recConnectionUpdateKernel.setArg(argIndex++, layer._recConnections);
			_recConnectionUpdateKernel.setArg(argIndex++, layer._recReconstruction); 
			_recConnectionUpdateKernel.setArg(argIndex++, layer._ffStates);
			_recConnectionUpdateKernel.setArg(argIndex++, layerSize);
			_recConnectionUpdateKernel.setArg(argIndex++, layerDesc._recRadius);
			_recConnectionUpdateKernel.setArg(argIndex++, layerDesc._ffAlpha);

			cs.getQueue().enqueueNDRangeKernel(_recConnectionUpdateKernel, cl::NullRange, cl::NDRange(layerDesc._width, layerDesc._height));
		}

		// FB
		{
			float sparsitySquared = layerDesc._sparsity * layerDesc._sparsity;

			cl_int2 nextSize;

			if (l == 0) {
				nextSize.x = _inputWidth;
				nextSize.y = _inputHeight;
			}
			else {
				nextSize.x = _layerDescs[l - 1]._width;
				nextSize.y = _layerDescs[l - 1]._height;
			}

			cl_float2 layerToInputsScalar = { static_cast<float>(layerSize.x) / static_cast<float>(nextSize.x),
				static_cast<float>(layerSize.y) / static_cast<float>(nextSize.y) };

			{
				int argIndex = 0;

				if (l == _layers.size() - 1) {
					_fbConnectionUpdateKernel.setArg(argIndex++, _layers[l]._ffStatesPrev);
					_fbConnectionUpdateKernel.setArg(argIndex++, _layers[l]._ffStates);
				}
				else {
					_fbConnectionUpdateKernel.setArg(argIndex++, _layers[l + 1]._fbStatesExploratoryPrev);
					_fbConnectionUpdateKernel.setArg(argIndex++, _layers[l + 1]._fbStatesExploratory);
				}

				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbConnectionsPrev);
				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbConnections);
				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbStatesPrev);
				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbStates);
				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbStatesExploratory);

				if (l == 0)
					_fbConnectionUpdateKernel.setArg(argIndex++, _inputs);
				else
					_fbConnectionUpdateKernel.setArg(argIndex++, _layers[l - 1]._ffStates);

				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbStates);
				_fbConnectionUpdateKernel.setArg(argIndex++, layer._fbStatesExploratory);
				_fbConnectionUpdateKernel.setArg(argIndex++, layerSize);	
				_fbConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbRadius);
				_fbConnectionUpdateKernel.setArg(argIndex++, layerToInputsScalar); 
				_fbConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbPredAlpha);
				_fbConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbRLAlpha);
				_fbConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbLambdaGamma);
				_fbConnectionUpdateKernel.setArg(argIndex++, rlError);

				cs.getQueue().enqueueNDRangeKernel(_fbConnectionUpdateKernel, cl::NullRange, cl::NDRange(nextSize.x, nextSize.y));
			}

			if (l != 0) {
				int argIndex = 0;

				_predConnectionUpdateKernel.setArg(argIndex++, layer._predConnectionsPrev);
				_predConnectionUpdateKernel.setArg(argIndex++, layer._predConnections);
				_predConnectionUpdateKernel.setArg(argIndex++, layer._fbStatesPrev);
				_predConnectionUpdateKernel.setArg(argIndex++, layer._fbStates);
				_predConnectionUpdateKernel.setArg(argIndex++, layer._fbStatesExploratory);
				_predConnectionUpdateKernel.setArg(argIndex++, _layers[l - 1]._ffStatesPrev);
				_predConnectionUpdateKernel.setArg(argIndex++, _layers[l - 1]._ffStates);
				_predConnectionUpdateKernel.setArg(argIndex++, layerSize);
				_predConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbRadius);
				_predConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbPredAlpha);
				_predConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbRLAlpha);
				_predConnectionUpdateKernel.setArg(argIndex++, layerDesc._fbLambdaGamma);
				_predConnectionUpdateKernel.setArg(argIndex++, rlError);

				cs.getQueue().enqueueNDRangeKernel(_predConnectionUpdateKernel, cl::NullRange, cl::NDRange(nextSize.x, nextSize.y));
			}
		}

		// Update prevs
		prevLayerWidth = layerDesc._width;
		prevLayerHeight = layerDesc._height;

		pPrevLayer = &layer._ffStates;
	}

	// Buffer swaps
	for (int l = 0; l < _layers.size(); l++) {
		Layer &layer = _layers[l];

		std::swap(layer._ffStates, layer._ffStatesPrev);

		std::swap(layer._fbStates, layer._fbStatesPrev);
		std::swap(layer._fbStatesExploratory, layer._fbStatesExploratoryPrev);

		std::swap(layer._ffConnections, layer._ffConnectionsPrev);
		std::swap(layer._recConnections, layer._recConnectionsPrev);
		std::swap(layer._fbConnections, layer._fbConnectionsPrev);
	}
}
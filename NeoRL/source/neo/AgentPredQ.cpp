#include "AgentPredQ.h"

#include <iostream>

using namespace neo;

void AgentPredQ::createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
	cl_int2 inputSize, cl_int2 actionSize, cl_int2 qSize,
	cl_int2 inputCoderSize, cl_int2 actionCoderSize, cl_int2 qCoderSize,
	const std::vector<LayerDesc> &layerDescs,
	cl_float2 initWeightRange,
	std::mt19937 &rng)
{
	_inputSize = inputSize;
	_actionSize = actionSize;
	_qSize = qSize;

	_inputCoderSize = inputCoderSize;
	_actionCoderSize = actionCoderSize;
	_qCoderSize = qCoderSize;

	_layerDescs = layerDescs;
	_layers.resize(_layerDescs.size());

	cl::Kernel randomUniform2DXYKernel = cl::Kernel(program.getProgram(), "randomUniform2DXY");
	cl::Kernel randomUniform2DXYZKernel = cl::Kernel(program.getProgram(), "randomUniform2DXYZ");

	cl_int2 prevLayerSize = inputSize;

	for (int l = 0; l < _layers.size(); l++) {
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs;

		if (l == 0) {
			scDescs.resize(3);

			scDescs[0]._size = _inputCoderSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[0]._useTraces = false;

			scDescs[1]._size = _actionCoderSize;
			scDescs[1]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[1]._ignoreMiddle = false;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[1]._useTraces = false;

			scDescs[2]._size = _qCoderSize;
			scDescs[2]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[2]._ignoreMiddle = false;
			scDescs[2]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[2]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[2]._useTraces = false;
		}
		else {
			scDescs.resize(2);

			scDescs[0]._size = prevLayerSize;
			scDescs[0]._radius = _layerDescs[l]._feedForwardRadius;
			scDescs[0]._ignoreMiddle = false;
			scDescs[0]._weightAlpha = _layerDescs[l]._scWeightAlpha;
			scDescs[0]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[0]._useTraces = true;

			scDescs[1]._size = _layerDescs[l]._size;
			scDescs[1]._radius = _layerDescs[l]._recurrentRadius;
			scDescs[1]._ignoreMiddle = true;
			scDescs[1]._weightAlpha = _layerDescs[l]._scWeightRecurrentAlpha;
			scDescs[1]._weightLambda = _layerDescs[l]._scWeightLambda;
			scDescs[1]._useTraces = true;
		}

		_layers[l]._sc.createRandom(cs, program, scDescs, _layerDescs[l]._size, _layerDescs[l]._lateralRadius, initWeightRange, rng);

		std::vector<Predictor::VisibleLayerDesc> predDescs;

		if (l < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[l]._size;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;

			predDescs[1]._size = _layerDescs[l]._size;
			predDescs[1]._radius = _layerDescs[l]._feedBackRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[l]._size;
			predDescs[0]._radius = _layerDescs[l]._predictiveRadius;
		}

		if (l == 0)
			_layers[l]._pred.createRandom(cs, program, predDescs, _actionSize, initWeightRange, true, rng);
		else
			_layers[l]._pred.createRandom(cs, program, predDescs, _layerDescs[l - 1]._size, initWeightRange, true, rng);

		// Create baselines
		_layers[l]._predRewardBaselines = createDoubleBuffer2D(cs, _layerDescs[l]._size, CL_R, CL_FLOAT);

		_layers[l]._predReward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);
		_layers[l]._propagatedPredReward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layerDescs[l]._size.x, _layerDescs[l]._size.y);

		cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

		cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
		cl::array<cl::size_type, 3> layerRegion = { _layerDescs[l]._size.x, _layerDescs[l]._size.y, 1 };

		cs.getQueue().enqueueFillImage(_layers[l]._predRewardBaselines[_back], zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._predReward, zeroColor, zeroOrigin, layerRegion);
		cs.getQueue().enqueueFillImage(_layers[l]._propagatedPredReward, zeroColor, zeroOrigin, layerRegion);

		prevLayerSize = _layerDescs[l]._size;
	}

	_qInput = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _qSize.x, _qSize.y);

	_qTransform = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGB, CL_FLOAT), _qSize.x, _qSize.y);

	// Q Predictor
	{
		std::vector<Predictor::VisibleLayerDesc> predDescs;

		if (0 < _layers.size() - 1) {
			predDescs.resize(2);

			predDescs[0]._size = _layerDescs[0]._size;
			predDescs[0]._radius = _layerDescs[0]._predictiveRadius;

			predDescs[1]._size = _layerDescs[0 + 1]._size;
			predDescs[1]._radius = _layerDescs[0]._feedBackRadius;
		}
		else {
			predDescs.resize(1);

			predDescs[0]._size = _layerDescs[0]._size;
			predDescs[0]._radius = _layerDescs[0]._predictiveRadius;
		}

		_qPred.createRandom(cs, program, predDescs, _qSize, initWeightRange, true, rng);
	}

	// Random Q transform
	randomUniformXYZ(_qTransform, cs, randomUniform2DXYZKernel, _qSize, { -1.0f, 1.0f }, rng);

	_inputWhitener.create(cs, program, _inputSize, CL_R, CL_FLOAT);
	_actionWhitener.create(cs, program, _actionSize, CL_R, CL_FLOAT);
	_qWhitener.create(cs, program, _qSize, CL_R, CL_FLOAT);

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
	_predictionRewardPropagationKernel = cl::Kernel(program.getProgram(), "phPredictionRewardPropagation");
	_setQKernel = cl::Kernel(program.getProgram(), "phSetQ");

	// Create coders
	{
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs(1);

		scDescs[0]._size = _inputSize;
		scDescs[0]._radius = _inputCoderFeedForwardRadius;
		scDescs[0]._ignoreMiddle = false;
		scDescs[0]._weightAlpha = _inputCoderAlpha;
		scDescs[0]._useTraces = false;

		_inputCoder.createRandom(cs, program, scDescs, _inputCoderSize, _inputCoderLateralRadius, initWeightRange, rng);
	}

	{
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs(1);

		scDescs[0]._size = _actionSize;
		scDescs[0]._radius = _inputCoderFeedForwardRadius;
		scDescs[0]._ignoreMiddle = false;
		scDescs[0]._weightAlpha = _actionCoderAlpha;
		scDescs[0]._useTraces = false;

		_actionCoder.createRandom(cs, program, scDescs, _actionCoderSize, _actionCoderLateralRadius, initWeightRange, rng);
	}

	{
		std::vector<ComparisonSparseCoder::VisibleLayerDesc> scDescs(1);

		scDescs[0]._size = _qSize;
		scDescs[0]._radius = _qCoderFeedForwardRadius;
		scDescs[0]._ignoreMiddle = false;
		scDescs[0]._weightAlpha = _qCoderAlpha;
		scDescs[0]._useTraces = false;

		_qCoder.createRandom(cs, program, scDescs, _qCoderSize, _qCoderLateralRadius, initWeightRange, rng);
	}
}

void AgentPredQ::simStep(sys::ComputeSystem &cs, const cl::Image2D &input, const cl::Image2D &actionTaken, float reward, std::mt19937 &rng, bool learn, bool whiten) {
	// Place previous Q into Q buffer
	{
		int argIndex = 0;

		_setQKernel.setArg(argIndex++, _qTransform);
		_setQKernel.setArg(argIndex++, _qInput);
		_setQKernel.setArg(argIndex++, _prevQ);

		cs.getQueue().enqueueNDRangeKernel(_setQKernel, cl::NullRange, cl::NDRange(_qSize.x, _qSize.y));
	}
	
	// Whiten input
	if (whiten)
		_inputWhitener.filter(cs, input, _whiteningKernelRadius, _whiteningIntensity);

	_actionWhitener.filter(cs, actionTaken, _whiteningKernelRadius, _whiteningIntensity);

	_qWhitener.filter(cs, _qInput, _whiteningKernelRadius, _whiteningIntensity);

	// Feed to coders
	{
		std::vector<cl::Image2D> visibleStates(1);

		visibleStates[0] = whiten ? _inputWhitener.getResult() : input;

		_inputCoder.activate(cs, visibleStates, _inputCoderActiveRatio);

		if (learn)
			_inputCoder.learn(cs, visibleStates, _inputCoderBoostAlpha, _inputCoderActiveRatio);
	}

	{
		std::vector<cl::Image2D> visibleStates(1);

		visibleStates[0] = _actionWhitener.getResult();

		_actionCoder.activate(cs, visibleStates, _actionCoderActiveRatio);

		if (learn)
			_actionCoder.learn(cs, visibleStates, _actionCoderBoostAlpha, _actionCoderActiveRatio);
	}

	{
		std::vector<cl::Image2D> visibleStates(1);

		visibleStates[0] = _qWhitener.getResult();

		_qCoder.activate(cs, visibleStates, _qCoderActiveRatio);

		if (learn)
			_qCoder.learn(cs, visibleStates, _qCoderBoostAlpha, _qCoderActiveRatio);
	}

	// Feed forward
	for (int l = 0; l < _layers.size(); l++) {
		{
			std::vector<cl::Image2D> visibleStates;

			if (l == 0) {
				visibleStates.resize(3);

				visibleStates[0] = _inputCoder.getHiddenStates()[_back];
				visibleStates[1] = _actionCoder.getHiddenStates()[_back];
				visibleStates[2] = _qCoder.getHiddenStates()[_back];
			}
			else {
				visibleStates.resize(2);

				visibleStates[0] = _layers[l - 1]._sc.getHiddenStates()[_back];
				visibleStates[1] = _layers[l]._sc.getHiddenStates()[_back];
			}

			_layers[l]._sc.activate(cs, visibleStates, _layerDescs[l]._scActiveRatio);

			// Get reward
			if (l < _layers.size() - 1) {
				int argIndex = 0;

				_predictionRewardKernel.setArg(argIndex++, _layers[l + 1]._pred.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._sc.getHiddenStates()[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predReward);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predRewardBaselines[_back]);
				_predictionRewardKernel.setArg(argIndex++, _layers[l]._predRewardBaselines[_front]);
				_predictionRewardKernel.setArg(argIndex++, _layerDescs[l]._scActiveRatio);
				_predictionRewardKernel.setArg(argIndex++, _layerDescs[l]._predRewardBaselineDecay);

				cs.getQueue().enqueueNDRangeKernel(_predictionRewardKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			// Propagate reward
			if (l != 0) {
				// Propagate to first target
				cl_float2 hiddenToVisible = cl_float2{ static_cast<float>(_layerDescs[l - 1]._size.x) / static_cast<float>(_layerDescs[l]._size.x),
					static_cast<float>(_layerDescs[l - 1]._size.y) / static_cast<float>(_layerDescs[l]._size.y)
				};

				cl_float2 visibleToHidden = cl_float2{ static_cast<float>(_layerDescs[l]._size.x) / static_cast<float>(_layerDescs[l - 1]._size.x),
					static_cast<float>(_layerDescs[l]._size.y) / static_cast<float>(_layerDescs[l - 1]._size.y)
				};

				cl_int radius = std::max(static_cast<int>(std::ceil(visibleToHidden.x * (_layerDescs[l]._predictiveRadius + 0.5f))), static_cast<int>(std::ceil(visibleToHidden.y * (_layerDescs[l]._predictiveRadius + 0.5f))));

				int argIndex = 0;

				_predictionRewardPropagationKernel.setArg(argIndex++, _layers[l - 1]._predReward);
				_predictionRewardPropagationKernel.setArg(argIndex++, _layers[l]._propagatedPredReward);
				_predictionRewardPropagationKernel.setArg(argIndex++, hiddenToVisible);
				_predictionRewardPropagationKernel.setArg(argIndex++, _layerDescs[l - 1]._size);
				_predictionRewardPropagationKernel.setArg(argIndex++, radius);

				cs.getQueue().enqueueNDRangeKernel(_predictionRewardPropagationKernel, cl::NullRange, cl::NDRange(_layerDescs[l]._size.x, _layerDescs[l]._size.y));
			}

			if (learn) {
				if (l == 0)
					_layers[l]._sc.learn(cs, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
				else
					_layers[l]._sc.learn(cs, _layers[l]._propagatedPredReward, visibleStates, _layerDescs[l]._scBoostAlpha, _layerDescs[l]._scActiveRatio);
			}
		}
	}

	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::vector<cl::Image2D> visibleStates;

		if (l < _layers.size() - 1) {
			visibleStates.resize(2);

			visibleStates[0] = _layers[l]._sc.getHiddenStates()[_back];
			visibleStates[1] = _layers[l + 1]._pred.getHiddenStates()[_back];
		}
		else {
			visibleStates.resize(1);

			visibleStates[0] = _layers[l]._sc.getHiddenStates()[_back];
		}

		_layers[l]._pred.activate(cs, visibleStates, l == 0 ? Predictor::_tanH : Predictor::_binary);
	}

	// Q predictor
	{
		std::vector<cl::Image2D> visibleStates;

		if (0 < _layers.size() - 1) {
			visibleStates.resize(2);

			visibleStates[0] = _layers[0]._sc.getHiddenStates()[_back];
			visibleStates[1] = _layers[0 + 1]._pred.getHiddenStates()[_back];
		}
		else {
			visibleStates.resize(1);

			visibleStates[0] = _layers[0]._sc.getHiddenStates()[_back];
		}

		_qPred.activate(cs, visibleStates, Predictor::_identity);
	}

	// Recover Q
	std::vector<float> qValues(_qSize.x * _qSize.y);

	cs.getQueue().enqueueReadImage(_qPred.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_qSize.x), static_cast<cl::size_type>(_qSize.y), 1 }, 0, 0, qValues.data());

	// Average all Q values
	float q = 0.0f;

	for (int i = 0; i < qValues.size(); i++)
		q += qValues[i];

	q /= qValues.size();

	// Bellman equation
	float tdError = reward + _qGamma * q - _prevValue;

	float newQ = _prevValue + _qAlpha * tdError;

	if (learn) {
		for (int l = _layers.size() - 1; l >= 0; l--) {
			std::vector<cl::Image2D> visibleStatesPrev;

			if (l < _layers.size() - 1) {
				visibleStatesPrev.resize(2);

				visibleStatesPrev[0] = _layers[l]._sc.getHiddenStates()[_front];
				visibleStatesPrev[1] = _layers[l + 1]._pred.getHiddenStates()[_front];
			}
			else {
				visibleStatesPrev.resize(1);

				visibleStatesPrev[0] = _layers[l]._sc.getHiddenStates()[_front];
			}

			if (l == 0)
				_layers[l]._pred.learn(cs, tdError, actionTaken, visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
			else
				_layers[l]._pred.learn(cs, tdError, _layers[l - 1]._sc.getHiddenStates()[_back], visibleStatesPrev, _layerDescs[l]._predWeightAlpha, _layerDescs[l]._predWeightLambda);
		}

		// Q Pred
		{
			std::vector<cl::Image2D> visibleStatesPrev;

			if (0 < _layers.size() - 1) {
				visibleStatesPrev.resize(2);

				visibleStatesPrev[0] = _layers[0]._sc.getHiddenStates()[_front];
				visibleStatesPrev[1] = _layers[0 + 1]._pred.getHiddenStates()[_front];
			}
			else {
				visibleStatesPrev.resize(1);

				visibleStatesPrev[0] = _layers[0]._sc.getHiddenStates()[_front];
			}

			_qPred.learnQ(cs, tdError, visibleStatesPrev, _qWeightAlpha, _qWeightLambda);
		}
	}

	std::cout << "Q: " << newQ << std::endl;

	_prevQ = newQ;
	_prevTDError = tdError;
	_prevValue = q;
}

void AgentPredQ::clearMemory(sys::ComputeSystem &cs) {
	// Fix me
	abort();
}

void AgentPredQ::writeToStream(sys::ComputeSystem &cs, std::ostream &os) const {
	abort(); // Not working yet

			 // Layer information
	os << _layers.size() << std::endl;

	for (int li = 0; li < _layers.size(); li++) {
		const Layer &l = _layers[li];
		const LayerDesc &ld = _layerDescs[li];

		// Desc
		os << ld._size.x << " " << ld._size.y << " " << ld._feedForwardRadius << " " << ld._recurrentRadius << " " << ld._lateralRadius << " " << ld._feedBackRadius << " " << ld._predictiveRadius << std::endl;
		//os << ld._scWeightAlpha << " " << ld._scWeightRecurrentAlpha << " " << ld._scWeightLambda << " " << ld._scActiveRatio << " " << ld._scBoostAlpha << std::endl;
		os << ld._predWeightAlpha << std::endl;

		//l._sc.writeToStream(cs, os);
		l._pred.writeToStream(cs, os);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			//cs.getQueue().enqueueReadImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());

			for (int ri = 0; ri < rewards.size(); ri++)
				os << rewards[ri] << " ";
		}

		os << std::endl;

		{
			std::vector<cl_float> hiddenStatesPrev(ld._size.x * ld._size.y);

			//cs.getQueue().enqueueReadImage(l._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, hiddenStatesPrev.data());

			for (int si = 0; si < hiddenStatesPrev.size(); si++)
				os << hiddenStatesPrev[si] << " ";
		}

		os << std::endl;
	}
}

void AgentPredQ::readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is) {
	abort(); // Not working yet

			 // Layer information
	int numLayers;

	is >> numLayers;

	_layers.resize(numLayers);
	_layerDescs.resize(numLayers);

	for (int li = 0; li < _layers.size(); li++) {
		Layer &l = _layers[li];
		LayerDesc &ld = _layerDescs[li];

		// Desc
		is >> ld._size.x >> ld._size.y >> ld._feedForwardRadius >> ld._recurrentRadius >> ld._lateralRadius >> ld._feedBackRadius >> ld._predictiveRadius;
		//is >> ld._scWeightAlpha >> ld._scWeightRecurrentAlpha >> ld._scWeightLambda >> ld._scActiveRatio >> ld._scBoostAlpha;
		is >> ld._predWeightAlpha;

		//l._reward = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		//l._scHiddenStatesPrev = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), ld._size.x, ld._size.y);

		//l._sc.readFromStream(cs, program, is);
		l._pred.readFromStream(cs, program, is);

		// Layer
		{
			std::vector<cl_float> rewards(ld._size.x * ld._size.y);

			for (int ri = 0; ri < rewards.size(); ri++)
				is >> rewards[ri];

			//cs.getQueue().enqueueWriteImage(l._reward, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, rewards.data());
		}

		{
			std::vector<cl_float> hiddenStatesPrev(ld._size.x * ld._size.y);

			for (int si = 0; si < hiddenStatesPrev.size(); si++)
				is >> hiddenStatesPrev[si];

			//cs.getQueue().enqueueWriteImage(l._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(ld._size.x), static_cast<cl::size_type>(ld._size.y), 1 }, 0, 0, hiddenStatesPrev.data());
		}
	}

	_predictionRewardKernel = cl::Kernel(program.getProgram(), "phPredictionReward");
}
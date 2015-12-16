#pragma once

#include "ComparisonSparseCoder.h"
#include "PredictorSwarm.h"

namespace neo {
	class AgentSPG {
	public:
		struct LayerDesc {
			cl_int2 _hiddenSize;

			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius;
			cl_int _qRadiusHiddenFeedForwardAttention, _qRadiusHiddenRecurrentAttention, _qRadiusHiddenAction, _qRadius;
			cl_int _startRadiusHiddenFeedForwardAttention, _startRadiusHiddenRecurrentAttention, _startRadiusHiddenAction;

			cl_float _scWeightAlpha;
			cl_float _scWeightRecurrentAlpha;
			cl_float _scWeightLambda;
			cl_float _scActiveRatio;
			cl_float _scBoostAlpha;

			cl_float _baseLineDecay;
			cl_float _baseLineSensitivity;

			cl_float3 _predWeightAlpha;
			cl_float2 _predWeightLambda;

			cl_float _gamma;

			cl_float _noise;

			cl_float _minAttention;

			LayerDesc()
				: _hiddenSize({ 8, 8 }),
				_feedForwardRadius(4), _recurrentRadius(4), _lateralRadius(4), _feedBackRadius(4), _predictiveRadius(4),
				_qRadiusHiddenFeedForwardAttention(4), _qRadiusHiddenRecurrentAttention(4), _qRadiusHiddenAction(4), _qRadius(4),
				_startRadiusHiddenFeedForwardAttention(4), _startRadiusHiddenRecurrentAttention(4), _startRadiusHiddenAction(4),
				_scWeightAlpha(0.001f), _scWeightRecurrentAlpha(0.0001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.1f), _scBoostAlpha(0.05f),
				_baseLineDecay(0.01f), _baseLineSensitivity(0.01f),
				_predWeightAlpha({ 0.1f, 0.001f, 0.01f }),
				_predWeightLambda({ 0.95f, 0.95f }),
				_gamma(0.99f), _noise(0.05f),
				_minAttention(0.05f)
			{}
		};

		struct Layer {
			ComparisonSparseCoder _sc;

			PredictorSwarm _predAction;
			PredictorSwarm _predAttentionFeedForward;
			PredictorSwarm _predAttentionRecurrent;

			cl::Image2D _modulatedFeedForwardInput;
			cl::Image2D _modulatedRecurrentInput;
			cl::Image2D _inhibitedAction;

			DoubleBuffer2D _baseLines;

			cl::Image2D _reward;

			cl::Image2D _scHiddenStatesPrev;
		};

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

		cl::Image2D _action;

		cl::Kernel _baseLineUpdateKernel;
		cl::Kernel _baseLineUpdateSumErrorKernel;
		cl::Kernel _inhibitKernel;
		cl::Kernel _modulateKernel;
		cl::Kernel _copyActionKernel;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerPredictorRadius, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange, float initThreshold,
			std::mt19937 &rng);

		void simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng);

		void clearMemory(sys::ComputeSystem &cs);

		size_t getNumLayers() const {
			return _layers.size();
		}

		const Layer &getLayer(int index) const {
			return _layers[index];
		}

		const LayerDesc &getLayerDescs(int index) const {
			return _layerDescs[index];
		}

		const cl::Image2D &getAction() const {
			return _action;
		}
	};
}
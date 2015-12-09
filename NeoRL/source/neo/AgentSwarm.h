#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"
#include "Swarm.h"

namespace neo {
	class AgentSwarm {
	public:
		struct LayerDesc {
			cl_int2 _hiddenSize;

			cl_int2 _qSize;

			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius;
			cl_int _qRadiusHiddenFeedForwardAttention, _qRadiusHiddenRecurrentAttention, _qRadiusHiddenAction, _qRadius;
			cl_int _startRadiusHiddenFeedForwardAttention, _startRadiusHiddenRecurrentAttention, _startRadiusHiddenAction;

			cl_float _scWeightAlpha;
			cl_float _scWeightLambda;
			cl_float _scActiveRatio;
			cl_float _scBoostAlpha;

			cl_float _baseLineDecay;
			cl_float _baseLineSensitivity;

			cl_float _predWeightAlpha;

			cl_int _swarmAnnealingIterations;
			cl_float _swarmActionDeriveAlpha;

			cl_float _swarmQAlpha;
			cl_float _swarmQHiddenAlpha;
			cl_float _swarmPredAlpha;
			cl_float _swarmLambda;
			cl_float _swarmGamma;

			cl_float _swarmExpPert;
			cl_float _swarmExpBreak;

			LayerDesc()
				: _hiddenSize({ 8, 8 }), _qSize({ 4, 4 }),
				_feedForwardRadius(1), _recurrentRadius(4), _lateralRadius(4), _feedBackRadius(4), _predictiveRadius(4),
				_qRadiusHiddenFeedForwardAttention(4), _qRadiusHiddenRecurrentAttention(4), _qRadiusHiddenAction(4), _qRadius(4),
				_startRadiusHiddenFeedForwardAttention(4), _startRadiusHiddenRecurrentAttention(4), _startRadiusHiddenAction(4),
				_scWeightAlpha(0.001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.05f), _scBoostAlpha(0.01f),
				_baseLineDecay(0.01f), _baseLineSensitivity(0.01f),
				_predWeightAlpha(0.05f),
				_swarmAnnealingIterations(1), _swarmActionDeriveAlpha(0.05f),
				_swarmQAlpha(0.001f), _swarmQHiddenAlpha(0.01f),
				_swarmPredAlpha(0.01f), _swarmLambda(0.95f), _swarmGamma(0.99f),
				_swarmExpPert(0.05f), _swarmExpBreak(0.01f)
			{}
		};

		struct Layer {
			ComparisonSparseCoder _sc;
			Predictor _pred;
			Swarm _swarm;

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

		cl::Image2D _lastLayerAction;

		Predictor _firstLayerPred;

		cl::Kernel _baseLineUpdateKernel;
		cl::Kernel _inhibitKernel;
		cl::Kernel _modulateKernel;

	public:
		cl_float _predWeightAlpha;

		AgentSwarm()
			: _predWeightAlpha(0.01f)
		{}

		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerPredictorRadius, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange, float initThreshold,
			std::mt19937 &rng);

		void simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn = true);

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

		const Predictor &getFirstLayerPred() const {
			return _firstLayerPred;
		}

		const cl::Image2D &getExploratoryActions() const {
			return _layers.front()._swarm.getVisibleLayer(1)._actionsExploratory;
		}
	};
}
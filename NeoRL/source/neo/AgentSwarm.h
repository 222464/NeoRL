#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"
#include "PredictorSwarm.h"

namespace neo {
	class AgentSwarm {
	public:
		struct LayerDesc {
			cl_int2 _size;

			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius;

			cl_int _scIterations;
			cl_float _scLeak;
			cl_float _scWeightAlpha;
			cl_float _scLateralWeightAlpha;
			cl_float _scThresholdAlpha;
			cl_float _scWeightTraceLambda;
			cl_float _scActiveRatio;

			cl_float _baseLineDecay;
			cl_float _baseLineSensitivity;

			cl_float3 _predWeightAlpha;
			cl_float2 _predWeightLambda;

			cl_float _explorationBreakChance;

			cl_float _gamma;
	
			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(5), _recurrentRadius(5), _lateralRadius(5), _feedBackRadius(5), _predictiveRadius(5),
				_scIterations(20), _scLeak(0.02f),
				_scWeightAlpha(0.001f), _scLateralWeightAlpha(0.1f), _scThresholdAlpha(0.005f),
				_scWeightTraceLambda(0.95f), _scActiveRatio(0.05f),
				_baseLineDecay(0.01f), _baseLineSensitivity(4.0f),
				_predWeightAlpha({ 0.005f, 0.001f, 0.01f }),
				_predWeightLambda({ 0.95f,0.95f }),
				_explorationBreakChance(0.05f),
				_gamma(0.99f)
			{}
		};

		struct Layer {
			ComparisonSparseCoder _sc;
			PredictorSwarm _pred;

			DoubleBuffer2D _baseLines;

			cl::Image2D _reward;

			cl::Image2D _scHiddenStatesPrev;
		};

	private:
		cl::Image2D _inputsImage;
		cl::Image2D _actionsImage;

		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

		Predictor _inputPred;
		PredictorSwarm _actionPred;

		std::vector<float> _inputs;
		std::vector<float> _actions; 
		std::vector<float> _inputPredictions;
		std::vector<cl_float2> _actionPredictions;

		cl::Kernel _baseLineUpdateKernel;

	public:
		cl_float3 _predWeightAlpha;
		cl_float2 _predWeightLambda;
		float _inputPredWeightAlpha;

		cl_float _gamma;

		cl_float _explorationStdDev;
		
		AgentSwarm()
			: _predWeightAlpha({ 0.005f, 0.001f, 0.01f }),
			_predWeightLambda({ 0.95f,0.95f }),
			_inputPredWeightAlpha(0.02f),
			_gamma(0.99f),
			_explorationStdDev(0.1f)
		{}

		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int inputPredictorRadius, cl_int actionPredictorRadius,
			cl_int actionFeedForwardRadius, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
			cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange, std::mt19937 &rng);

		void simStep(float reward, sys::ComputeSystem &cs, std::mt19937 &rng, bool learn = true);

		void setState(int index, float value) {
			_inputs[index] = value;
		}

		void setState(int x, int y, float value) {
			setState(x + y * _layers.front()._sc.getVisibleLayerDesc(0)._size.x, value);
		}

		float getAction(int index) const {
			return _actions[index];
		}

		float getAction(int x, int y) const {
			return getAction(x + y * _layers.front()._sc.getVisibleLayerDesc(1)._size.x);
		}

		float getPrediction(int index) const {
			return _inputPredictions[index];
		}

		float getPrediction(int x, int y) const {
			return getPrediction(x + y * _layers.front()._sc.getVisibleLayerDesc(0)._size.x);
		}

		size_t getNumLayers() const {
			return _layers.size();
		}

		const Layer &getLayer(int index) const {
			return _layers[index];
		}

		const LayerDesc &getLayerDescs(int index) const {
			return _layerDescs[index];
		}

		const Predictor &getInputPred() const {
			return _inputPred;
		}

		const PredictorSwarm &getActionPred() const {
			return _actionPred;
		}
	};
}
#pragma once

#include "SparseCoder.h"
#include "Predictor.h"

namespace neo {
	class AgentQRoute {
	public:
		enum InputType {
			_state, _action
		};

		struct QConnection {
			float _weight;

			float _trace;

			QConnection()
				: _trace(0.0f)
			{}
		};
		struct LayerDesc {
			cl_int2 _size;

			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius, _qRadius;

			cl_int _scIterations;
			cl_float _scLeak;
			cl_float _scWeightAlpha;
			cl_float _scLateralWeightAlpha;
			cl_float _scThresholdAlpha;
			cl_float _scWeightTraceLambda;
			cl_float _scActiveRatio;

			cl_float _baseLineDecay;
			cl_float _baseLineSensitivity;

			cl_float _predWeightAlpha;

			cl_float _qAlpha;
			cl_float _qGammaLambda;
			cl_float _qReluLeak;

			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(4), _recurrentRadius(4), _lateralRadius(4), _feedBackRadius(4), _predictiveRadius(4), _qRadius(4),
				_scIterations(17), _scLeak(0.1f),
				_scWeightAlpha(0.01f), _scLateralWeightAlpha(0.05f), _scThresholdAlpha(0.01f),
				_scWeightTraceLambda(0.95f), _scActiveRatio(0.02f),
				_baseLineDecay(0.01f), _baseLineSensitivity(4.0f),
				_predWeightAlpha(0.1f),
				_qAlpha(0.005f), _qGammaLambda(0.95f), _qReluLeak(0.01f)
			{}
		};

		struct Layer {
			SparseCoder _sc;
			Predictor _pred;

			DoubleBuffer2D _baseLines;

			cl::Image2D _reward;

			cl::Image2D _scHiddenStatesPrev;

			// Q
			DoubleBuffer2D _qStates;
			DoubleBuffer3D _qWeights;
			DoubleBuffer2D _qBiases;
			cl::Image2D _qErrorTemp;
		};

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

		std::vector<QConnection> _qConnections;
		std::vector<float> _qStates;
		std::vector<float> _qErrors;

		std::vector<float> _prediction;

		std::vector<float> _inputLayerStates;
		std::vector<float> _qInputLayerErrors;
		std::vector<InputType> _inputTypes;

		cl_float _prevValue;

		Predictor _firstLayerPred;

		cl::Kernel _baseLineUpdateKernel;

		cl::Kernel _qForwardKernel;
		cl::Kernel _qBackwardKernel;
		cl::Kernel _qBackwardFirstLayerKernel;
		cl::Kernel _qWeightUpdateKernel;

		cl::Image2D _input;
		cl::Image2D _lastLayerError;
		cl::Image2D _inputLayerError;

	public:
		cl_float _predWeightAlpha;
		cl_int _qIter;
		cl_float _actionDeriveAlpha;
		cl_float _lastLayerQAlpha;
		cl_float _lastLayerQGammaLambda;
		cl_float _lasyLayerQReluLeak;

		cl_float _gamma;

		cl_float _explorationPerturbationStdDev;
		cl_float _explorationBreakChance;

		AgentQRoute()
			: _predWeightAlpha(0.1f),
			_qIter(10),
			_actionDeriveAlpha(0.04f),
			_lastLayerQAlpha(0.005f), _lastLayerQGammaLambda(0.95f),
			_lasyLayerQReluLeak(0.01f),
			_gamma(0.99f),
			_explorationPerturbationStdDev(0.05f), _explorationBreakChance(0.01f),
			_prevValue(0.0f)
		{}

		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int firstLayerPredictorRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
			cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange, std::mt19937 &rng);

		void simStep(sys::ComputeSystem &cs, float reward, std::mt19937 &rng, bool learn = true);

		void setInput(int index, float value) {
			_inputLayerStates[index] = value;
		}

		float getAction(int index) const {
			return _inputLayerStates[index];
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

		const Predictor &getFirstLayerPred() const {
			return _firstLayerPred;
		}
	};
}
#pragma once

#include "IRSDR.h"

namespace sdr {
	class IPredictiveRSDR {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius, _recurrentRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent;

			float _learnFeedBack, _learnPrediction;

			int _sdrIter;
			float _sdrStepSize;
			float _sdrLambda;
			float _sdrHiddenDecay;
			float _sdrWeightDecay;
			float _sdrBoostSparsity;
			float _sdrLearnBoost;
			float _sdrNoise;

			float _averageSurpriseDecay;
			float _attentionFactor;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(8), _recurrentRadius(6), _predictiveRadius(6), _feedBackRadius(8),
				_learnFeedForward(0.1f), _learnRecurrent(0.1f),
				_learnFeedBack(0.1f), _learnPrediction(0.1f),
				_sdrIter(30), _sdrStepSize(0.1f), _sdrLambda(0.4f), _sdrHiddenDecay(0.01f), _sdrWeightDecay(0.0001f),
				_sdrBoostSparsity(0.1f), _sdrLearnBoost(0.005f), _sdrNoise(0.1f),
				_averageSurpriseDecay(0.01f),
				_attentionFactor(2.0f)
			{}
		};

		struct PredictionNode {
			std::vector<Connection> _feedBackConnections;
			std::vector<Connection> _predictiveConnections;

			Connection _bias;

			float _state;
			float _statePrev;

			float _activation;
			float _activationPrev;

			float _averageSurprise; // Use to keep track of importance for prediction. If current error is greater than average, then attention is > 0.5 else < 0.5 (sigmoid)

			PredictionNode()
				: _state(0.0f), _statePrev(0.0f), _activation(0.0f), _activationPrev(0.0f), _averageSurprise(0.0f)
			{}
		};

		struct Layer {
			IRSDR _sdr;

			std::vector<PredictionNode> _predictionNodes;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<PredictionNode> _inputPredictionNodes;


	public:
		float _learnInputFeedBack;

		IPredictiveRSDR()
			: _learnInputFeedBack(0.1f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initBoost, std::mt19937 &generator);

		void simStep(std::mt19937 &generator, bool learn = true);

		void setInput(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value);
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getPrediction(int index) const {
			return _inputPredictionNodes[index]._state;
		}

		float getPrediction(int x, int y) const {
			return getPrediction(x + y * _layers.front()._sdr.getVisibleWidth());
		}

		const std::vector<LayerDesc> &getLayerDescs() const {
			return _layerDescs;
		}

		const std::vector<Layer> &getLayers() const {
			return _layers;
		}
	};
}
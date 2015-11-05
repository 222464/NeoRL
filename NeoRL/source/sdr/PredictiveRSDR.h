#pragma once

#include "RSDR.h"

namespace sdr {
	class PredictiveRSDR {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
		};

		struct LayerDesc {
			int _width, _height;

			int _receptiveRadius, _recurrentRadius, _lateralRadius, _predictiveRadius, _feedBackRadius;

			float _learnFeedForward, _learnRecurrent, _learnLateral, _learnThreshold;

			float _learnFeedBack, _learnPrediction;

			int _subIterSettle;
			int _subIterMeasure;
			float _leak;

			float _averageSurpriseDecay;
			float _attentionFactor;

			float _sparsity;

			LayerDesc()
				: _width(16), _height(16),
				_receptiveRadius(8), _recurrentRadius(4), _lateralRadius(3), _predictiveRadius(4), _feedBackRadius(8),
				_learnFeedForward(0.02f), _learnRecurrent(0.02f), _learnLateral(0.2f), _learnThreshold(0.12f),
				_learnFeedBack(0.05f), _learnPrediction(0.05f),
				_subIterSettle(17), _subIterMeasure(5), _leak(0.1f),
				_averageSurpriseDecay(0.01f),
				_attentionFactor(4.0f),
				_sparsity(0.01f)
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
			RSDR _sdr;

			std::vector<PredictionNode> _predictionNodes;
		};

		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}
	
	private:
		std::vector<LayerDesc> _layerDescs;
		std::vector<Layer> _layers;

		std::vector<float> _prediction;

	public:
		void createRandom(int inputWidth, int inputHeight, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator);

		void simStep(bool learn = true);

		void setInput(int index, float value) {
			_layers.front()._sdr.setVisibleState(index, value);
		}

		void setInput(int x, int y, float value) {
			setInput(x + y * _layers.front()._sdr.getVisibleWidth(), value);
		}

		float getPrediction(int index) const {
			return _prediction[index];
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
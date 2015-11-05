#include "IPRSDRRL.h"

#include <SFML/Window.hpp>
#include <iostream>

#include <algorithm>

using namespace sdr;

void IPRSDRRL::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<InputType> &inputTypes, const std::vector<LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initBoost, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_inputTypes = inputTypes;

	for (int i = 0; i < _inputTypes.size(); i++) {
		if (_inputTypes[i] == _action)
			_actionInputIndices.push_back(i);
	}

	_layerDescs = layerDescs;

	_layers.resize(_layerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < _layerDescs.size(); l++) {
		_layers[l]._sdr.createRandom(widthPrev, heightPrev, _layerDescs[l]._width, _layerDescs[l]._height, _layerDescs[l]._receptiveRadius, _layerDescs[l]._recurrentRadius, initMinWeight, initMaxWeight, initBoost, generator);

		_layers[l]._predictionNodes.resize(_layerDescs[l]._width * _layerDescs[l]._height);

		int feedBackSize = std::pow(_layerDescs[l]._feedBackRadius * 2 + 1, 2);
		int predictiveSize = std::pow(_layerDescs[l]._predictiveRadius * 2 + 1, 2);

		float hiddenToNextHiddenWidth = 1.0f;
		float hiddenToNextHiddenHeight = 1.0f;

		if (l < _layers.size() - 1) {
			hiddenToNextHiddenWidth = static_cast<float>(_layerDescs[l + 1]._width) / static_cast<float>(_layerDescs[l]._width);
			hiddenToNextHiddenHeight = static_cast<float>(_layerDescs[l + 1]._height) / static_cast<float>(_layerDescs[l]._height);
		}

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._bias._weightQ = weightDist(generator);
			p._bias._weightPrediction = weightDist(generator);
			//p._bias._weightAction = weightDist(generator);

			int hx = pi % _layerDescs[l]._width;
			int hy = pi / _layerDescs[l]._width;

			// Feed Back
			if (l < _layers.size() - 1) {
				p._feedBackConnections.reserve(feedBackSize);

				int centerX = std::round(hx * hiddenToNextHiddenWidth);
				int centerY = std::round(hy * hiddenToNextHiddenHeight);

				for (int dx = -_layerDescs[l]._feedBackRadius; dx <= _layerDescs[l]._feedBackRadius; dx++)
					for (int dy = -_layerDescs[l]._feedBackRadius; dy <= _layerDescs[l]._feedBackRadius; dy++) {
						int hox = centerX + dx;
						int hoy = centerY + dy;

						if (hox >= 0 && hox < _layerDescs[l + 1]._width && hoy >= 0 && hoy < _layerDescs[l + 1]._height) {
							int hio = hox + hoy * _layerDescs[l + 1]._width;

							Connection c;

							c._weightQ = weightDist(generator);
							c._weightPrediction = weightDist(generator);
							//c._weightAction = weightDist(generator);
							c._index = hio;

							p._feedBackConnections.push_back(c);
						}
					}

				p._feedBackConnections.shrink_to_fit();
			}

			// Predictive
			p._predictiveConnections.reserve(feedBackSize);

			for (int dx = -_layerDescs[l]._predictiveRadius; dx <= _layerDescs[l]._predictiveRadius; dx++)
				for (int dy = -_layerDescs[l]._predictiveRadius; dy <= _layerDescs[l]._predictiveRadius; dy++) {
					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < _layerDescs[l]._width && hoy >= 0 && hoy < _layerDescs[l]._height) {
						int hio = hox + hoy * _layerDescs[l]._width;

						Connection c;

						c._weightQ = weightDist(generator);
						c._weightPrediction = weightDist(generator);
						//c._weightAction = weightDist(generator);
						c._index = hio;

						p._predictiveConnections.push_back(c);
					}
				}

			p._predictiveConnections.shrink_to_fit();
		}

		widthPrev = _layerDescs[l]._width;
		heightPrev = _layerDescs[l]._height;
	}

	_inputPredictionNodes.resize(_inputTypes.size());

	float inputToNextHiddenWidth = static_cast<float>(_layerDescs.front()._width) / static_cast<float>(inputWidth);
	float inputToNextHiddenHeight = static_cast<float>(_layerDescs.front()._height) / static_cast<float>(inputHeight);

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputNode &p = _inputPredictionNodes[pi];

		p._bias._weightQ = weightDist(generator);
		p._bias._weightPrediction = weightDist(generator);
		//p._bias._weightAction = weightDist(generator);

		int hx = pi % inputWidth;
		int hy = pi / inputWidth;

		int feedBackSize = std::pow(inputFeedBackRadius * 2 + 1, 2);

		// Feed Back
		p._feedBackConnections.reserve(feedBackSize);

		int centerX = std::round(hx * inputToNextHiddenWidth);
		int centerY = std::round(hy * inputToNextHiddenHeight);

		for (int dx = -inputFeedBackRadius; dx <= inputFeedBackRadius; dx++)
			for (int dy = -inputFeedBackRadius; dy <= inputFeedBackRadius; dy++) {
				int hox = centerX + dx;
				int hoy = centerY + dy;

				if (hox >= 0 && hox < _layerDescs.front()._width && hoy >= 0 && hoy < _layerDescs.front()._height) {
					int hio = hox + hoy * _layerDescs.front()._width;

					Connection c;

					c._weightQ = weightDist(generator);
					c._weightPrediction = weightDist(generator);
					//c._weightAction = weightDist(generator);
					c._index = hio;

					p._feedBackConnections.push_back(c);
				}
			}

		p._feedBackConnections.shrink_to_fit();
	}
}

void IPRSDRRL::simStep(float reward, std::mt19937 &generator) {
	// Feature extraction
	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.activate(_layerDescs[l]._sdrIter, _layerDescs[l]._sdrStepSize, _layerDescs[l]._sdrLambda, _layerDescs[l]._sdrHiddenDecay, _layerDescs[l]._sdrNoise, generator);

		// Set inputs for next layer if there is one
		if (l < _layers.size() - 1) {
			for (int i = 0; i < _layers[l]._sdr.getNumHidden(); i++)
				_layers[l + 1]._sdr.setVisibleState(i, _layers[l]._sdr.getHiddenState(i));
		}
	}

	// Prediction
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (int l = _layers.size() - 1; l >= 0; l--) {
		std::normal_distribution<float> pertDist(0.0f, _layerDescs[l]._exploratoryNoise);
		
		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			float prediction = 0.0f;
			float action = 0.0f;
			float q = 0.0f;

			// Feed Back
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
					prediction += p._feedBackConnections[ci]._weightPrediction * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
					//action += p._feedBackConnections[ci]._weightAction * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._actionExploratory;
					q += p._feedBackConnections[ci]._weightQ * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
				}
			}

			// Predictive
			for (int ci = 0; ci < p._predictiveConnections.size(); ci++) {
				prediction += p._predictiveConnections[ci]._weightPrediction * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);
				//action += p._predictiveConnections[ci]._weightAction * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);
				q += p._predictiveConnections[ci]._weightQ * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);
			}

			// Threshold
			p._prediction = sigmoid(prediction) * 2.0f - 1.0f;

			p._q = q;

			float tdError = reward + _layerDescs[l]._gamma * p._q - p._qPrev;

			float learnPrediction = tdError > 0.0f ? 1.0f : _layerDescs[l]._predictionDrift;

			float predictionError = _layers[l]._sdr.getHiddenState(pi) - p._predictionPrev;

			// Update Q and action traces and weights
			if (l < _layers.size() - 1) {
				for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
					// Action
					p._feedBackConnections[ci]._weightPrediction += _layerDescs[l]._learnFeedBackPred * learnPrediction * p._feedBackConnections[ci]._tracePrediction;

					p._feedBackConnections[ci]._tracePrediction = _layerDescs[l]._gammaLambda * p._feedBackConnections[ci]._tracePrediction + predictionError * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
				
					// Q
					p._feedBackConnections[ci]._weightQ += _layerDescs[l]._learnFeedBackQ * tdError * p._feedBackConnections[ci]._traceQ;

					p._feedBackConnections[ci]._traceQ = _layerDescs[l]._gammaLambda * p._feedBackConnections[ci]._traceQ + _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
				}
			}

			for (int ci = 0; ci < p._predictiveConnections.size(); ci++) {
				// Action
				p._predictiveConnections[ci]._weightPrediction += _layerDescs[l]._learnPredictionPred * tdError * p._predictiveConnections[ci]._tracePrediction;

				p._predictiveConnections[ci]._tracePrediction = _layerDescs[l]._gammaLambda * p._predictiveConnections[ci]._tracePrediction + predictionError * _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);
			
				// Q
				p._predictiveConnections[ci]._weightQ += _layerDescs[l]._learnFeedBackQ * tdError * p._predictiveConnections[ci]._traceQ;

				p._predictiveConnections[ci]._traceQ = _layerDescs[l]._gammaLambda * p._predictiveConnections[ci]._traceQ + _layers[l]._sdr.getHiddenState(p._predictiveConnections[ci]._index);
			}
		}
	}
	
	// Get first layer prediction
	{
		std::normal_distribution<float> pertDist(0.0f, _exploratoryNoise);

		for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
			InputNode &p = _inputPredictionNodes[pi];

			float prediction = 0.0f;
			float action = 0.0f;
			float q = 0.0f;

			// Feed Back
			for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
				prediction += p._feedBackConnections[ci]._weightPrediction * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
				//action += p._feedBackConnections[ci]._weightAction * _layers[l + 1]._predictionNodes[p._feedBackConnections[ci]._index]._actionExploratory;
				q += p._feedBackConnections[ci]._weightQ * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
			}

			// Threshold
			p._prediction = sigmoid(prediction) * 2.0f - 1.0f;

			p._predictionExploratory = p._prediction;

			if (_inputTypes[pi] == _action)
				p._predictionExploratory = std::min(1.0f, std::max(-1.0f, p._prediction + pertDist(generator)));

			p._q = q;

			float tdError = reward + _gamma * p._q - p._qPrev;

			float learnPrediction = tdError > 0.0f ? 1.0f : _predictionDrift;

			float predictionError = _layers.front()._sdr.getVisibleState(pi) - p._predictionPrev;

			// Update Q and action traces and weights
			for (int ci = 0; ci < p._feedBackConnections.size(); ci++) {
				// Action
				p._feedBackConnections[ci]._weightPrediction += _learnFeedBackPred * learnPrediction * p._feedBackConnections[ci]._tracePrediction;

				p._feedBackConnections[ci]._tracePrediction = _gammaLambda * p._feedBackConnections[ci]._tracePrediction + predictionError * _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._prediction;

				// Q
				p._feedBackConnections[ci]._weightQ += _learnFeedBackQ * tdError * p._feedBackConnections[ci]._traceQ;

				p._feedBackConnections[ci]._traceQ = _gammaLambda * p._feedBackConnections[ci]._traceQ + _layers.front()._predictionNodes[p._feedBackConnections[ci]._index]._prediction;
			}
		}
	}

	for (int l = 0; l < _layers.size(); l++) {
		_layers[l]._sdr.learn(_layerDescs[l]._learnFeedForward, _layerDescs[l]._learnRecurrent, _layerDescs[l]._sdrLearnBoost, _layerDescs[l]._sdrBoostSparsity, _layerDescs[l]._sdrWeightDecay, _layerDescs[l]._sdrMaxWeightDelta);

		_layers[l]._sdr.stepEnd();

		for (int pi = 0; pi < _layers[l]._predictionNodes.size(); pi++) {
			PredictionNode &p = _layers[l]._predictionNodes[pi];

			p._predictionPrev = p._prediction;
			p._qPrev = p._q;
		}
	}

	for (int pi = 0; pi < _inputPredictionNodes.size(); pi++) {
		InputNode &p = _inputPredictionNodes[pi];

		p._predictionPrev = p._prediction;
		p._qPrev = p._q;
	}

	// Set action inputs to actions
	for (int i = 0; i < _inputTypes.size(); i++)
		if (_inputTypes[i] == _action)
			_layers.front()._sdr.setVisibleState(i, getAction(i));
}
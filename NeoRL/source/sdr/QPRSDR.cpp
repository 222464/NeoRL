#include "QPRSDR.h"

#include <iostream>

using namespace sdr;

void QPRSDR::createRandom(int inputWidth, int inputHeight, const std::vector<int> &actionIndices, const std::vector<PredictiveRSDR::LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);
	
	_prsdr.createRandom(inputWidth, inputHeight, layerDescs, initMinWeight, initMaxWeight, initThreshold, generator);

	_actionNodeIndices.resize(inputWidth * inputHeight);

	for (int i = 0; i < _actionNodeIndices.size(); i++)
		_actionNodeIndices[i] = -1;

	for (int i = 0; i < actionIndices.size(); i++)
		_actionNodeIndices[actionIndices[i]] = i;

	_qFunctionLayers.resize(layerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < layerDescs.size(); l++) {
		_qFunctionLayers[l]._qFunctionNodes.resize(layerDescs[l]._width * layerDescs[l]._height);

		_qFunctionLayers[l]._qConnections.resize(layerDescs[l]._width * layerDescs[l]._height);

		for (int i = 0; i < _qFunctionLayers[l]._qConnections.size(); i++)
			_qFunctionLayers[l]._qConnections[i]._weight = weightDist(generator);

		int feedForwardSize = std::pow(layerDescs[l]._receptiveRadius * 2 + 1, 2);

		float hiddenToPrevHiddenWidth = static_cast<float>(widthPrev) / static_cast<float>(layerDescs[l]._width);
		float hiddenToPrevHiddenHeight = static_cast<float>(heightPrev) / static_cast<float>(layerDescs[l]._height);

		for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
			QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

			q._bias._weight = weightDist(generator);

			int hx = qi % layerDescs[l]._width;
			int hy = qi / layerDescs[l]._width;

			// Feed Forward
			q._feedForwardConnections.reserve(feedForwardSize);

			int centerX = std::round(hx * hiddenToPrevHiddenWidth);
			int centerY = std::round(hy * hiddenToPrevHiddenHeight);

			for (int dx = -layerDescs[l]._receptiveRadius; dx <= layerDescs[l]._receptiveRadius; dx++)
				for (int dy = -layerDescs[l]._receptiveRadius; dy <= layerDescs[l]._receptiveRadius; dy++) {
					int hox = centerX + dx;
					int hoy = centerY + dy;

					if (hox >= 0 && hox < widthPrev && hoy >= 0 && hoy < heightPrev) {
						int hio = hox + hoy * widthPrev;

						Connection c;

						c._weight = weightDist(generator);
						c._index = hio;

						if (l == 0) {
							if (_actionNodeIndices[hio] != -1)
								q._feedForwardConnections.push_back(c);
						}
						else {
							if (!_qFunctionLayers[l - 1]._qFunctionNodes[c._index]._feedForwardConnections.empty())
								q._feedForwardConnections.push_back(c);
						}
					}
				}

			q._feedForwardConnections.shrink_to_fit();
		}

		widthPrev = layerDescs[l]._width;
		heightPrev = layerDescs[l]._height;
	}
	
	_actionNodes.resize(actionIndices.size());

	for (int i = 0; i < _actionNodes.size(); i++)
		_actionNodes[i]._inputIndex = actionIndices[i];
}

void QPRSDR::simStep(float reward, std::mt19937 &generator, bool learn) {
	_prsdr.simStep(learn);

	// Starting action is predicted action
	for (int i = 0; i < _actionNodes.size(); i++)
		_actionNodes[i]._predictedAction = _actionNodes[i]._deriveAction = _prsdr.getPrediction(_actionNodes[i]._inputIndex);

	// Derive action
	for (int iter = 0; iter < _actionDeriveIterations; iter++) {
		// Feed forward
		for (int l = 0; l < _qFunctionLayers.size(); l++) {
			if (l > 0) {
				int prevLayerIndex = l - 1;

				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					float sum = 0.0f;// q._bias._weight;

					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
						sum += q._feedForwardConnections[ci]._weight *_qFunctionLayers[prevLayerIndex]._qFunctionNodes[q._feedForwardConnections[ci]._index]._state;

					q._state = sigmoid(sum) * _prsdr.getLayers()[l]._predictionNodes[qi]._state;

					// Zero error for later
					q._error = 0.0f;
				}
			}
			else {
				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					float sum = 0.0f;// q._bias._weight;

					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
						sum += q._feedForwardConnections[ci]._weight * _actionNodes[_actionNodeIndices[q._feedForwardConnections[ci]._index]]._deriveAction;

					q._state = sigmoid(sum) * _prsdr.getLayers()[l]._predictionNodes[qi]._state;

					// Zero error for later
					q._error = 0.0f;
				}
			}
		}

		// Zero action error for later
		for (int i = 0; i < _actionNodes.size(); i++)
			_actionNodes[i]._error = 0.0f;

		// Final Q layer
		float q = 0.0f;

		for (int l = 0; l < _qFunctionLayers.size(); l++) {
			for (int i = 0; i < _qFunctionLayers.back()._qFunctionNodes.size(); i++) {
				q += _qFunctionLayers[l]._qConnections[i]._weight * _qFunctionLayers[l]._qFunctionNodes[i]._state;
			}
		}

		std::cout << q << std::endl;

		// Backpropagate positive Q error
		for (int l = _qFunctionLayers.size() - 1; l >= 0; l--) {
			// Last layer
			for (int i = 0; i < _qFunctionLayers[l]._qFunctionNodes.size(); i++)
				_qFunctionLayers[l]._qFunctionNodes[i]._error += _qFunctionLayers[l]._qConnections[i]._weight;// *relud(_qFunctionLayers.back()._qFunctionNodes[i]._state, _reluLeak);

			if (l > 0) {
				int prevLayerIndex = l - 1;

				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					// Find complete error for this node
					q._error = q._error * q._state * (1.0f - q._state);// (q._state, _reluLeak);

					// Propagate error to other nodes
					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
						_qFunctionLayers[prevLayerIndex]._qFunctionNodes[q._feedForwardConnections[ci]._index]._error += q._error * q._feedForwardConnections[ci]._weight;
				}
			}
			else {
				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					// Find complete error for this node
					q._error = q._error * q._state * (1.0f - q._state);

					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
						_actionNodes[_actionNodeIndices[q._feedForwardConnections[ci]._index]]._error += q._error * q._feedForwardConnections[ci]._weight;
				}
			}
		}

		// Update derive action
		for (int i = 0; i < _actionNodes.size(); i++)
			_actionNodes[i]._deriveAction = std::min(1.0f, std::max(0.0f, _actionNodes[i]._deriveAction + (_actionNodes[i]._error > 0.0f ? 1.0f : -1.0f) * _actionDeriveAlpha));
	}

	// Explore
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> pertDist(0.0f, _explorationStdDev);

	for (int i = 0; i < _actionNodes.size(); i++) {
		if (dist01(generator) < _explorationBreak)
			_actionNodes[i]._exploratoryAction = dist01(generator);
		else
			_actionNodes[i]._exploratoryAction = std::min(1.0f, std::max(0.0f, _actionNodes[i]._deriveAction + pertDist(generator)));
	}

	// Final feed-forward pass to calculate Q
	for (int l = 0; l < _qFunctionLayers.size(); l++) {
		if (l > 0) {
			int prevLayerIndex = l - 1;

			for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
				QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

				float sum = 0.0f;// q._bias._weight;

				for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
					sum += q._feedForwardConnections[ci]._weight *_qFunctionLayers[prevLayerIndex]._qFunctionNodes[q._feedForwardConnections[ci]._index]._state;

				q._state = sigmoid(sum) * _prsdr.getLayers()[l]._predictionNodes[qi]._state;

				// Zero erro again
				q._error = 0.0f;
			}
		}
		else {
			for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
				QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

				float sum = 0.0f;// q._bias._weight;

				for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
					sum += q._feedForwardConnections[ci]._weight * _actionNodes[_actionNodeIndices[q._feedForwardConnections[ci]._index]]._exploratoryAction;

				q._state = sigmoid(sum) * _prsdr.getLayers()[l]._predictionNodes[qi]._state;

				// Zero error again
				q._error = 0.0f;
			}
		}
	}

	float q = 0.0f;

	for (int l = 0; l < _qFunctionLayers.size(); l++) {
		for (int i = 0; i < _qFunctionLayers.back()._qFunctionNodes.size(); i++) {
			q += _qFunctionLayers[l]._qConnections[i]._weight * _qFunctionLayers[l]._qFunctionNodes[i]._state;
		}
	}

	float tdError = reward + _gamma * q - _prevValue;

	float qAlphaTdError = _qAlpha * tdError;
	float actionAlphaTdError = _actionAlpha * tdError;

	//std::cout << q << std::endl;

	_prevValue = q;

	// Update weights
	if (learn) {
		// Backpropagate positive error
		for (int l = _qFunctionLayers.size() - 1; l >= 0; l--) {
			for (int i = 0; i < _qFunctionLayers[l]._qFunctionNodes.size(); i++)
				_qFunctionLayers[l]._qFunctionNodes[i]._error += _qFunctionLayers[l]._qConnections[i]._weight;// *relud(_qFunctionLayers.back()._qFunctionNodes[i]._state, _reluLeak);

			if (l > 0) {
				int prevLayerIndex = l - 1;

				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					// Find complete error for this node
					q._error = q._error * q._state * (1.0f - q._state);

					// Propagate error to other nodes
					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
						_qFunctionLayers[prevLayerIndex]._qFunctionNodes[q._feedForwardConnections[ci]._index]._error += q._error * q._feedForwardConnections[ci]._weight;
				}
			}
			else {
				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					// Find complete error for this node
					q._error = q._error * q._state * (1.0f - q._state);

					//for (int ci = 0; ci < q._feedForwardConnections.size(); ci++)
					//	_actionNodes[_actionNodeIndices[q._feedForwardConnections[ci]._index]]._error += q._error * q._feedForwardConnections[ci]._weight;
				}
			}
		}

		// Update weights and traces
		for (int l = 0; l < _qFunctionLayers.size(); l++) {
			if (l > 0) {
				int prevLayerIndex = l - 1;

				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					q._bias._weight += actionAlphaTdError * q._bias._trace;

					q._bias._trace = _gammaLambda * q._bias._trace + q._error;

					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++) {
						q._feedForwardConnections[ci]._weight += actionAlphaTdError * q._feedForwardConnections[ci]._trace;
							
						q._feedForwardConnections[ci]._trace = _gammaLambda * q._feedForwardConnections[ci]._trace + q._error * _qFunctionLayers[prevLayerIndex]._qFunctionNodes[q._feedForwardConnections[ci]._index]._state;
					}
				}
			}
			else {
				for (int qi = 0; qi < _qFunctionLayers[l]._qFunctionNodes.size(); qi++) {
					QFunctionNode &q = _qFunctionLayers[l]._qFunctionNodes[qi];

					q._bias._weight += actionAlphaTdError * q._bias._trace;

					q._bias._trace = _gammaLambda * q._bias._trace + q._error;

					for (int ci = 0; ci < q._feedForwardConnections.size(); ci++) {
						q._feedForwardConnections[ci]._weight += actionAlphaTdError * q._feedForwardConnections[ci]._trace;

						q._feedForwardConnections[ci]._trace = _gammaLambda * q._feedForwardConnections[ci]._trace + q._error * _actionNodes[_actionNodeIndices[q._feedForwardConnections[ci]._index]]._exploratoryAction;
					}
				}
			}

			// Q connections
			for (int i = 0; i < _qFunctionLayers[l]._qConnections.size(); i++) {
				_qFunctionLayers[l]._qConnections[i]._weight += qAlphaTdError * _qFunctionLayers[l]._qConnections[i]._trace;

				_qFunctionLayers[l]._qConnections[i]._trace = _gammaLambda * _qFunctionLayers[l]._qConnections[i]._trace + _qFunctionLayers[l]._qFunctionNodes[i]._state;
			}
		}
	}
}
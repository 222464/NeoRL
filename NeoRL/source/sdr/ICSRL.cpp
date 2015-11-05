#include "ICSRL.h"

#include <iostream>

using namespace sdr;

void ICSRL::createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<ActionLayerDesc> &actionLayerDescs, const std::vector<IPredictiveRSDR::LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, float initBoost, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_prsdr.createRandom(inputWidth, inputHeight, inputFeedBackRadius, layerDescs, initMinWeight, initMaxWeight, initThreshold, generator);

	_actionLayerDescs = actionLayerDescs;

	_actionLayers.resize(_actionLayerDescs.size());

	int widthPrev = inputWidth;
	int heightPrev = inputHeight;

	for (int l = 0; l < _actionLayerDescs.size(); l++) {
		_actionLayers[l]._actionNodes.resize(layerDescs[l]._width * layerDescs[l]._height);

		int feedBackSize = std::pow(layerDescs[l]._feedBackRadius * 2 + 1, 2);

		float hiddenToNextHiddenWidth = static_cast<float>(layerDescs[l]._width) / static_cast<float>(widthPrev);
		float hiddenToNextHiddenHeight = static_cast<float>(layerDescs[l]._height) / static_cast<float>(heightPrev);

		for (int ai = 0; ai < _actionLayers[l]._actionNodes.size(); ai++) {
			ActionNode &a = _actionLayers[l]._actionNodes[ai];

			int hx = ai % widthPrev;
			int hy = ai / widthPrev;

			// Feed Back
			a._feedBackConnectionIndices.reserve(feedBackSize);

			int centerX = std::round(hx * hiddenToNextHiddenWidth);
			int centerY = std::round(hy * hiddenToNextHiddenHeight);

			for (int dx = -layerDescs[l]._feedBackRadius; dx <= layerDescs[l]._feedBackRadius; dx++)
				for (int dy = -layerDescs[l]._feedBackRadius; dy <= layerDescs[l]._feedBackRadius; dy++) {
					int hox = centerX + dx;
					int hoy = centerY + dy;

					if (hox >= 0 && hox < layerDescs[l]._width && hoy >= 0 && hoy < layerDescs[l]._height) {
						int hio = hox + hoy * layerDescs[l]._width;

						a._feedBackConnectionIndices.push_back(hio);
					}
				}

			a._feedBackConnectionIndices.shrink_to_fit();

			// Lateral
			a._lateralConnectionIndices.reserve(feedBackSize);

			for (int dx = -layerDescs[l]._predictiveRadius; dx <= layerDescs[l]._predictiveRadius; dx++)
				for (int dy = -layerDescs[l]._predictiveRadius; dy <= layerDescs[l]._predictiveRadius; dy++) {
					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < widthPrev && hoy >= 0 && hoy < heightPrev) {
						int hio = hox + hoy * widthPrev;

						a._lateralConnectionIndices.push_back(hio);
					}
				}

			a._lateralConnectionIndices.shrink_to_fit();

			a._sdrrl.createRandom(a._feedBackConnectionIndices.size() * (l == _actionLayers.size() - 1 ? 1 : 2) + l == 0 ? 0 : a._lateralConnectionIndices.size(), 2, _actionLayerDescs[l]._cellCount, initMinWeight, initMaxWeight, initThreshold, generator);
		}

		widthPrev = layerDescs[l]._width;
		heightPrev = layerDescs[l]._height;
	}
}

void ICSRL::simStep(float reward, std::mt19937 &generator, bool learn) {
	_prsdr.simStep(generator, learn);

	// Update action nodes
	for (int l = _actionLayers.size() - 1; l >= 0; l--) {
		for (int ai = 0; ai < _actionLayers[l]._actionNodes.size(); ai++) {
			ActionNode &a = _actionLayers[l]._actionNodes[ai];

			int inputIndex = 0;

			for (int ci = 0; ci < a._feedBackConnectionIndices.size(); ci++) {
				a._sdrrl.setState(inputIndex++, _prsdr.getLayers()[l]._predictionNodes[a._feedBackConnectionIndices[ci]]._state);
				//a._sdrrl.setState(inputIndex++, _actionLayers[l + 1][a._feedBackConnectionIndices[ci]]._state);
			}
		}
	}
}
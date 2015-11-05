#include "RSDR.h"

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sdr;

void RSDR::createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int inhibitionRadius, int recurrentRadius, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;

	_receptiveRadius = receptiveRadius;
	_inhibitionRadius = inhibitionRadius;
	_recurrentRadius = recurrentRadius;

	int numVisible = visibleWidth * visibleHeight;
	int numHidden = hiddenWidth * hiddenHeight;
	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
	int inhibitionSize = std::pow(inhibitionRadius * 2 + 1, 2);
	int recurrentSize = std::pow(recurrentRadius * 2 + 1, 2);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	float hiddenToVisibleWidth = static_cast<float>(visibleWidth) / static_cast<float>(hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(visibleHeight) / static_cast<float>(hiddenHeight);

	for (int hi = 0; hi < numHidden; hi++) {
		int hx = hi % hiddenWidth;
		int hy = hi / hiddenWidth;

		int centerX = std::round(hx * hiddenToVisibleWidth);
		int centerY = std::round(hy * hiddenToVisibleHeight);

		_hidden[hi]._threshold = initThreshold;

		// Receptive
		_hidden[hi]._feedForwardConnections.reserve(receptiveSize);

		for (int dx = -receptiveRadius; dx <= receptiveRadius; dx++)
			for (int dy = -receptiveRadius; dy <= receptiveRadius; dy++) {
				int vx = centerX + dx;
				int vy = centerY + dy;

				if (vx >= 0 && vx < visibleWidth && vy >= 0 && vy < visibleHeight) {
					int vi = vx + vy * visibleWidth;

					Connection c;

					c._weight = weightDist(generator);
					c._index = vi;
					
					_hidden[hi]._feedForwardConnections.push_back(c);
				}
			}

		_hidden[hi]._feedForwardConnections.shrink_to_fit();

		// Inhibition
		_hidden[hi]._lateralConnections.reserve(inhibitionSize);

		for (int dx = -inhibitionRadius; dx <= inhibitionRadius; dx++)
			for (int dy = -inhibitionRadius; dy <= inhibitionRadius; dy++) {
				if (dx == 0 && dy == 0)
					continue;

				int hox = hx + dx;
				int hoy = hy + dy;

				if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenHeight) {
					int hio = hox + hoy * hiddenWidth;

					//Connection c;

					//c._weight = inhibitionDist(generator);
					//c._index = hio;
		
					_hidden[hi]._lateralConnections.push_back(hio);
				}
			}

		_hidden[hi]._lateralConnections.shrink_to_fit();

		// Recurrent
		if (recurrentRadius != -1) {
			_hidden[hi]._recurrentConnections.reserve(recurrentSize);

			for (int dx = -recurrentRadius; dx <= recurrentRadius; dx++)
				for (int dy = -recurrentRadius; dy <= recurrentRadius; dy++) {
					if (dx == 0 && dy == 0)
						continue;

					int hox = hx + dx;
					int hoy = hy + dy;

					if (hox >= 0 && hox < hiddenWidth && hoy >= 0 && hoy < hiddenHeight) {
						int hio = hox + hoy * hiddenWidth;

						Connection c;

						c._weight = weightDist(generator);
						c._index = hio;

						_hidden[hi]._recurrentConnections.push_back(c);
					}
				}

			_hidden[hi]._recurrentConnections.shrink_to_fit();
		}
	}
}

void RSDR::activate(float sparsity) {
	// Activate
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = -_hidden[hi]._threshold;

		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			sum += _visible[_hidden[hi]._feedForwardConnections[ci]._index]._input * _hidden[hi]._feedForwardConnections[ci]._weight;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			sum += _hidden[_hidden[hi]._recurrentConnections[ci]._index]._statePrev * _hidden[hi]._recurrentConnections[ci]._weight;

		_hidden[hi]._activation = sum;
	}

	// Inhibit
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float numActive = sparsity * _hidden[hi]._lateralConnections.size();

		float inhibition = 0.0f;

		for (int ci = 0; ci < _hidden[hi]._lateralConnections.size(); ci++)
			inhibition += _hidden[_hidden[hi]._lateralConnections[ci]]._activation >= _hidden[hi]._activation ? 1.0f : 0.0f;

		_hidden[hi]._state = inhibition < numActive ? 1.0f : 0.0f;
	}
}

void RSDR::inhibit(float sparsity, const std::vector<float> &activations, std::vector<float> &states) {
	states.clear();
	states.assign(_hidden.size(), 0.0f);

	// Inhibit
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float numActive = sparsity * _hidden[hi]._lateralConnections.size();
		
		float inhibition = 0.0f;

		for (int ci = 0; ci < _hidden[hi]._lateralConnections.size(); ci++)
			inhibition += activations[_hidden[hi]._lateralConnections[ci]] >= activations[hi] ? 1.0f : 0.0f;

		states[hi] = inhibition < numActive ? 1.0f : 0.0f;
	}
}

void RSDR::reconstruct() {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			_visible[_hidden[hi]._feedForwardConnections[ci]._index]._reconstruction += _hidden[hi]._feedForwardConnections[ci]._weight * _hidden[hi]._state;

			visibleDivs[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._state;
		}

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++) {
			_hidden[_hidden[hi]._recurrentConnections[ci]._index]._reconstruction += _hidden[hi]._recurrentConnections[ci]._weight * _hidden[hi]._state;
	
			hiddenDivs[_hidden[hi]._recurrentConnections[ci]._index] += _hidden[hi]._state;
		}
	}

	//for (int vi = 0; vi < _visible.size(); vi++)
	//	_visible[vi]._reconstruction /= std::max(1.0f, visibleDivs[vi]);

	//for (int hi = 0; hi < _hidden.size(); hi++)
	//	_hidden[hi]._reconstruction /= std::max(1.0f, hiddenDivs[hi]);
}

void RSDR::reconstructFeedForward(const std::vector<float> &states, std::vector<float> &recon) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);

	recon.clear();
	recon.assign(_visible.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			recon[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];

			visibleDivs[_hidden[hi]._feedForwardConnections[ci]._index] += states[hi];
		}
	}

	//for (int vi = 0; vi < _visible.size(); vi++)
	//	recon[vi] /= std::max(1.0f, visibleDivs[vi]);
}

void RSDR::learn(float learnFeedForward, float learnRecurrent, float learnLateral, float learnThreshold, float sparsity) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	float sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float learn = _hidden[hi]._state;

		if (learn > 0.0f) {
			for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
				_hidden[hi]._feedForwardConnections[ci]._weight += learnFeedForward * learn * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index];// (_visible[_hidden[hi]._feedForwardConnections[ci]._index]._input - learn * _hidden[hi]._feedForwardConnections[ci]._weight);

			for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
				_hidden[hi]._recurrentConnections[ci]._weight += learnRecurrent * learn * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index];// (_hidden[_hidden[hi]._recurrentConnections[ci]._index]._statePrev - learn * _hidden[hi]._recurrentConnections[ci]._weight);
		}

		_hidden[hi]._threshold += learnThreshold * (_hidden[hi]._state - sparsity);
	}
}

void RSDR::learn(const std::vector<float> &attentions, float learnFeedForward, float learnRecurrent, float learnLateral, float learnThreshold, float sparsity) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	float sparsitySquared = sparsity * sparsity;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float learn = _hidden[hi]._state;

		if (learn > 0.0f) {
			for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
				_hidden[hi]._feedForwardConnections[ci]._weight += learnFeedForward * attentions[hi] * learn * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index];// (_visible[_hidden[hi]._feedForwardConnections[ci]._index]._input - learn * _hidden[hi]._feedForwardConnections[ci]._weight);

			for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
				_hidden[hi]._recurrentConnections[ci]._weight += learnRecurrent * attentions[hi] * learn * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index];// (_hidden[_hidden[hi]._recurrentConnections[ci]._index]._statePrev - learn * _hidden[hi]._recurrentConnections[ci]._weight);
		}

		_hidden[hi]._threshold += learnThreshold * (_hidden[hi]._state - sparsity);
	}
}

void RSDR::getVHWeights(int hx, int hy, std::vector<float> &rectangle) const {
	float hiddenToVisibleWidth = static_cast<float>(_visibleWidth - 1) / static_cast<float>(_hiddenWidth - 1);
	float hiddenToVisibleHeight = static_cast<float>(_visibleHeight - 1) / static_cast<float>(_hiddenHeight - 1);

	int dim = _receptiveRadius * 2 + 1;

	rectangle.resize(dim * dim, 0.0f);

	int hi = hx + hy * _hiddenWidth;

	int centerX = std::round(hx * hiddenToVisibleWidth);
	int centerY = std::round(hy * hiddenToVisibleHeight);

	for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
		int index = _hidden[hi]._feedForwardConnections[ci]._index;

		int vx = index % _visibleWidth;
		int vy = index / _visibleWidth;

		int dx = vx - centerX;
		int dy = vy - centerY;

		int rx = dx + _receptiveRadius;
		int ry = dy + _receptiveRadius;

		rectangle[rx + ry * dim] = _hidden[hi]._feedForwardConnections[ci]._weight;
	}
}

void RSDR::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._statePrev = _hidden[hi]._state;
}
#include "IRSDR.h"

#include <algorithm>

#include <SFML/System.hpp>
#include <SFML/Window.hpp>

#include <algorithm>

#include <assert.h>

#include <iostream>

using namespace sdr;

void IRSDR::createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int recurrentRadius, float initMinWeight, float initMaxWeight, float initBoost, std::mt19937 &generator) {
	std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

	_visibleWidth = visibleWidth;
	_visibleHeight = visibleHeight;
	_hiddenWidth = hiddenWidth;
	_hiddenHeight = hiddenHeight;

	_receptiveRadius = receptiveRadius;
	_recurrentRadius = recurrentRadius;

	int numVisible = visibleWidth * visibleHeight;
	int numHidden = hiddenWidth * hiddenHeight;
	int receptiveSize = std::pow(receptiveRadius * 2 + 1, 2);
	int recurrentSize = std::pow(recurrentRadius * 2 + 1, 2);

	_visible.resize(numVisible);

	_hidden.resize(numHidden);

	float hiddenToVisibleWidth = static_cast<float>(visibleWidth) / static_cast<float>(hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(visibleHeight) / static_cast<float>(hiddenHeight);

	for (int hi = 0; hi < numHidden; hi++) {
		int hx = hi % hiddenWidth;
		int hy = hi / hiddenWidth;

		_hidden[hi]._boost = initBoost;

		int centerX = std::round(hx * hiddenToVisibleWidth);
		int centerY = std::round(hy * hiddenToVisibleHeight);

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

void IRSDR::pL(const std::vector<float> &states, float stepSize, float lambda, float hiddenDecay) {
	reconstruct();

	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	// Activate - deltaH = alpha * (D * (x - Dh) - lambda * h / (sqrt(h^2 + e)))
	for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = 0.0f;

		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			sum += visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] * _hidden[hi]._feedForwardConnections[ci]._weight;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			sum += hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] * _hidden[hi]._recurrentConnections[ci]._weight;

		//-lambda * _hidden[hi]._state / std::sqrt(_hidden[hi]._state * _hidden[hi]._state + epsilon)
		//_hidden[hi]._state += stepSize * (sum - lambda * _hidden[hi]._state / std::sqrt(_hidden[hi]._state * _hidden[hi]._state + epsilon)) - hiddenDecay * _hidden[hi]._state;

		_hidden[hi]._state = states[hi] + stepSize * sum - hiddenDecay * states[hi];

		_hidden[hi]._state = std::max(std::abs(_hidden[hi]._state) - stepSize * _hidden[hi]._boost, 0.0f) * (_hidden[hi]._state > 0.0f ? 1.0f : -1.0f);
	
		_hidden[hi]._state = std::min(1.0f, std::max(-1.0f, _hidden[hi]._state));
	}
}

void IRSDR::activate(int iter, float stepSize, float lambda, float hiddenDecay, float noise, std::mt19937 &generator) {
	std::vector<float> y(_hidden.size());
	std::vector<float> t(_hidden.size());
	std::vector<float> tPrev(_hidden.size(), 0.0f);
	std::vector<float> xPrev(_hidden.size(), 0.0f);

	std::normal_distribution<float> noiseDist(0.0f, noise);

	/*for (int hi = 0; hi < _hidden.size(); hi++) {
		float sum = 0.0f;

		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			sum += _visible[_hidden[hi]._feedForwardConnections[ci]._index]._input * _hidden[hi]._feedForwardConnections[ci]._weight;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			sum += _hidden[_hidden[hi]._recurrentConnections[ci]._index]._statePrev * _hidden[hi]._recurrentConnections[ci]._weight;

		y[hi] = _hidden[hi]._state = sum + noiseDist(generator);
	}*/

	for (int hi = 0; hi < _hidden.size(); hi++) {
		y[hi] = (_hidden[hi]._state += noiseDist(generator));
	}

	for (int i = 0; i < iter; i++) {
		pL(y, stepSize, lambda, hiddenDecay);

		for (int hi = 0.0f; hi < t.size(); hi++)
			t[hi] = 0.5f * (1.0f + std::sqrt(1.0f + 4.0f * tPrev[hi] * tPrev[hi]));

		for (int hi = 0.0f; hi < y.size(); hi++)
			y[hi] = _hidden[hi]._state + (tPrev[hi] - 1.0f) / t[hi] * (_hidden[hi]._state - xPrev[hi]);

		tPrev = t;

		for (int hi = 0.0f; hi < xPrev.size(); hi++)
			xPrev[hi] = _hidden[hi]._state;
	}

	pL(y, stepSize, lambda, hiddenDecay);

	reconstruct();
}

void IRSDR::reconstruct() {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._reconstruction = 0.0f;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			_visible[_hidden[hi]._feedForwardConnections[ci]._index]._reconstruction += _hidden[hi]._feedForwardConnections[ci]._weight * _hidden[hi]._state;

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			_hidden[_hidden[hi]._recurrentConnections[ci]._index]._reconstruction += _hidden[hi]._recurrentConnections[ci]._weight * _hidden[hi]._state;
	}
}

void IRSDR::reconstruct(const std::vector<float> &states, std::vector<float> &reconHidden, std::vector<float> &reconVisible) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);
	std::vector<float> hiddenDivs(_hidden.size(), 0.0f);

	reconVisible.clear();
	reconVisible.assign(_visible.size(), 0.0f);

	reconHidden.clear();
	reconHidden.assign(_hidden.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			reconVisible[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			reconHidden[_hidden[hi]._recurrentConnections[ci]._index] += _hidden[hi]._recurrentConnections[ci]._weight * states[hi];
	}
}

void IRSDR::reconstructFeedForward(const std::vector<float> &states, std::vector<float> &recon) {
	std::vector<float> visibleDivs(_visible.size(), 0.0f);

	recon.clear();
	recon.assign(_visible.size(), 0.0f);

	for (int hi = 0; hi < _hidden.size(); hi++) {
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			recon[_hidden[hi]._feedForwardConnections[ci]._index] += _hidden[hi]._feedForwardConnections[ci]._weight * states[hi];
	}
}

void IRSDR::learn(float learnFeedForward, float learnRecurrent, float learnBoost, float boostSparsity, float weightDecay, float maxWeightDelta) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		float learn = _hidden[hi]._state;

		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++) {
			float delta = learnFeedForward * learn * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index] - weightDecay * _hidden[hi]._feedForwardConnections[ci]._weight;

			_hidden[hi]._feedForwardConnections[ci]._weight += std::min(maxWeightDelta, std::max(-maxWeightDelta, delta));
		}

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++) {
			float delta = learnRecurrent * learn * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index] - weightDecay * _hidden[hi]._recurrentConnections[ci]._weight;

			_hidden[hi]._recurrentConnections[ci]._weight += std::min(maxWeightDelta, std::max(-maxWeightDelta, delta));
		}

		_hidden[hi]._boost = std::max(0.0f, _hidden[hi]._boost + ((_hidden[hi]._state == 0.0f ? 0.0f : 1.0f) - boostSparsity) * learnBoost);
	}

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
		for (int hi = 0; hi < _hidden.size(); hi++)
			std::cout << hiddenErrors[hi] << " ";

		std::cout << std::endl;
	}
}

/*void IRSDR::learn(const std::vector<float> &attentions, float learnFeedForward, float learnRecurrent) {
	std::vector<float> visibleErrors(_visible.size(), 0.0f);
	std::vector<float> hiddenErrors(_hidden.size(), 0.0f);

	float error = 0.0f;

	for (int vi = 0; vi < _visible.size(); vi++)
		error += std::pow(_visible[vi]._input - _visible[vi]._reconstruction, 2);

	for (int hi = 0; hi < _hidden.size(); hi++)
		error += std::pow(_hidden[hi]._statePrev - _hidden[hi]._reconstruction, 2);

	std::cout << error << std::endl;

	for (int vi = 0; vi < _visible.size(); vi++)
		visibleErrors[vi] = _visible[vi]._input - _visible[vi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++)
		hiddenErrors[hi] = _hidden[hi]._statePrev - _hidden[hi]._reconstruction;

	for (int hi = 0; hi < _hidden.size(); hi++) {
		//if (_hidden[hi]._activation != 0.0f)
		for (int ci = 0; ci < _hidden[hi]._feedForwardConnections.size(); ci++)
			_hidden[hi]._feedForwardConnections[ci]._weight += learnFeedForward * _hidden[hi]._state * attentions[hi] * visibleErrors[_hidden[hi]._feedForwardConnections[ci]._index];

		for (int ci = 0; ci < _hidden[hi]._recurrentConnections.size(); ci++)
			_hidden[hi]._recurrentConnections[ci]._weight += learnRecurrent * _hidden[hi]._state * attentions[hi] * hiddenErrors[_hidden[hi]._recurrentConnections[ci]._index];
	}
}*/

void IRSDR::getVHWeights(int hx, int hy, std::vector<float> &rectangle) const {
	float hiddenToVisibleWidth = static_cast<float>(_visibleWidth) / static_cast<float>(_hiddenWidth);
	float hiddenToVisibleHeight = static_cast<float>(_visibleHeight) / static_cast<float>(_hiddenHeight);

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

void IRSDR::stepEnd() {
	for (int hi = 0; hi < _hidden.size(); hi++)
		_hidden[hi]._statePrev = _hidden[hi]._state;
}
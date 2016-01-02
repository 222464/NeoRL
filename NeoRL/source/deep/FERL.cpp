#include "FERL.h"

#include <algorithm>

#include <iostream>

using namespace deep;

FERL::FERL()
	: _zInv(1.0f), _prevValue(0.0f)
{}

void FERL::createRandom(int numState, int numAction, int numHidden, float weightStdDev, std::mt19937 &generator) {
	_numState = numState;
	_numAction = numAction;

	_visible.resize(numState + numAction);

	_hidden.resize(numHidden);

	_actions.resize(numAction);

	std::normal_distribution<float> weightDist(0.0f, weightStdDev);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight = weightDist(generator);

	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight = weightDist(generator);

		_hidden[k]._connections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight = weightDist(generator);
	}

	for (int a = 0; a < _actions.size(); a++) {
		_actions[a]._bias._weight = weightDist(generator);

		_actions[a]._connections.resize(_hidden.size());

		for (int vi = 0; vi < _hidden.size(); vi++)
			_actions[a]._connections[vi]._weight = weightDist(generator);
	}

	_zInv = 1.0f / std::sqrt(static_cast<float>(numState + numHidden));

	_prevVisible.clear();
	_prevVisible.assign(_visible.size(), 0.0f);

	_prevHidden.clear();
	_prevHidden.assign(_hidden.size(), 0.0f);
}

void FERL::createFromParents(const FERL &parent1, const FERL &parent2, float averageChance, std::mt19937 &generator) {
	_numState = parent1._numState;
	_numAction = parent1._numAction;

	_visible.resize(_numState + _numAction + parent1._hidden.size());

	_hidden.resize(parent1._hidden.size());

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight = uniformDist(generator) < averageChance ? (parent1._visible[vi]._bias._weight + parent2._visible[vi]._bias._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._visible[vi]._bias._weight : parent2._visible[vi]._bias._weight);

	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight = uniformDist(generator) < averageChance ? (parent1._hidden[k]._bias._weight + parent2._hidden[k]._bias._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._hidden[k]._bias._weight : parent2._hidden[k]._bias._weight);

		_hidden[k]._connections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight = uniformDist(generator) < averageChance ? (parent1._hidden[k]._connections[vi]._weight + parent2._hidden[k]._connections[vi]._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._hidden[k]._connections[vi]._weight : parent2._hidden[k]._connections[vi]._weight);
	}

	for (int a = 0; a < _actions.size(); a++) {
		_actions[a]._bias._weight = uniformDist(generator) < averageChance ? (parent1._actions[a]._bias._weight + parent2._actions[a]._bias._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._actions[a]._bias._weight : parent2._actions[a]._bias._weight);

		_actions[a]._connections.resize(_hidden.size());

		for (int vi = 0; vi < _hidden.size(); vi++)
			_actions[a]._connections[vi]._weight = uniformDist(generator) < averageChance ? (parent1._actions[a]._connections[vi]._weight + parent2._actions[a]._connections[vi]._weight) * 0.5f : (uniformDist(generator) < 0.5f ? parent1._actions[a]._connections[vi]._weight : parent2._actions[a]._connections[vi]._weight);
	}

	_zInv = 1.0f / std::sqrt(static_cast<float>(_numState + _hidden.size()));

	_prevVisible.clear();
	_prevVisible.assign(_visible.size(), 0.0f);
}

void FERL::mutate(float perturbationStdDev, std::mt19937 &generator) {
	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight += perturbationDist(generator);

	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight += perturbationDist(generator);

		_hidden[k]._connections.resize(_visible.size());

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight += perturbationDist(generator);
	}

	for (int a = 0; a < _actions.size(); a++) {
		_actions[a]._bias._weight += perturbationDist(generator);

		_actions[a]._connections.resize(_hidden.size());

		for (int vi = 0; vi < _hidden.size(); vi++)
			_actions[a]._connections[vi]._weight += perturbationDist(generator);
	}
}

float FERL::freeEnergy() const {
	float sum = 0.0f;

	for (int k = 0; k < _hidden.size(); k++) {
		sum -= _hidden[k]._bias._weight * _hidden[k]._state;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum -= _hidden[k]._connections[vi]._weight * _visible[vi]._state * _hidden[k]._state;

		//sum += _hidden[k]._state * std::log(_hidden[k]._state) + (1.0f - _hidden[k]._state) * std::log(1.0f - _hidden[k]._state);
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		sum -= _visible[vi]._bias._weight * _visible[vi]._state;

	return sum;
}

void FERL::step(const std::vector<float> &state, std::vector<float> &action,
	float reward, float qAlpha, float gamma, float lambdaGamma,
	float actionAlpha, int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha,
	float breakChance, float perturbationStdDev,
	int maxNumReplaySamples, int replayIterations, float gradientAlpha,
	std::mt19937 &generator)
{
	for (int i = 0; i < _numState; i++)
		_visible[i]._state = state[i];

	std::vector<float> maxAction(_numAction);

	float nextQ = -999999.0f;

	std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

	for (int s = 0; s < actionSearchSamples; s++) {
		// Start with previous best inputs on first iteration
		if (s == 0) {
			// Find last best action
			for (int a = 0; a < _actions.size(); a++) {
				float sum = _actions[a]._bias._weight;

				for (int k = 0; k < _prevHidden.size(); k++)
					sum += _actions[a]._connections[k]._weight * _prevHidden[k];

				_visible[a + _numState]._state = std::min(1.0f, std::max(-1.0f, sum));
			}
		}
		else { // Start with random inputs on other iterations		
			for (int j = 0; j < _numAction; j++)
				_visible[j + _numState]._state = uniformDist(generator) * 2.0f - 1.0f;
		}

		// Find best action and associated Q value
		for (int p = 0; p < actionSearchIterations; p++) {
			for (int j = 0; j < _numAction; j++) {
				int index = j + _numState;

				float sum = _visible[index]._bias._weight;

				for (int k = 0; k < _hidden.size(); k++)
					sum += _hidden[k]._connections[index]._weight * _hidden[k]._state;

				_visible[index]._state = std::min(1.0f, std::max(-1.0f, _visible[index]._state + actionSearchAlpha * sum));
			}

			// Q value of best action
			activate();
		}

		float q = value();

		if (q > nextQ) {
			nextQ = q;

			for (int j = 0; j < _numAction; j++)
				maxAction[j] = _visible[j + _numState]._state;
		}
	}

	// Actual action (perturbed from maximum)
	if (action.size() != _numAction)
		action.resize(_numAction);

	std::normal_distribution<float> perturbationDist(0.0f, perturbationStdDev);

	for (int j = 0; j < _numAction; j++)
		if (uniformDist(generator) < breakChance)
			action[j] = uniformDist(generator) * 2.0f - 1.0f;
		else
			action[j] = std::min(1.0f, std::max(-1.0f, maxAction[j] + perturbationDist(generator)));

	// Activate current (selected) action so eligibilities can be updated properly
	for (int j = 0; j < _numAction; j++)
		_visible[_numState + j]._state = action[j];

	activate();

	float predictedQ = value();

	// Update Q
	float tdError = reward + gamma * predictedQ - _prevValue;

	ReplaySample sample;

	sample._visible = _prevVisible;
	sample._q = _prevValue + qAlpha * tdError;
	sample._originalQ = _prevValue;

	// Update previous samples
	float g = gamma;

	for (std::list<ReplaySample>::iterator it = _replaySamples.begin(); it != _replaySamples.end(); it++) {
		it->_q += qAlpha * g * tdError;

		g *= gamma;
	}

	_replaySamples.push_front(sample);

	while (_replaySamples.size() > maxNumReplaySamples)
		_replaySamples.pop_back();

	// Create buffer for random sample access
	std::vector<ReplaySample*> pReplaySamples(_replaySamples.size());

	int replayIndex = 0;

	for (std::list<ReplaySample>::iterator it = _replaySamples.begin(); it != _replaySamples.end(); it++) {
		pReplaySamples[replayIndex] = &(*it);

		replayIndex++;
	}

	// Update on the chain
	std::uniform_int_distribution<int> replayDist(0, pReplaySamples.size() - 1);

	for (int r = 0; r < replayIterations; r++) {
		replayIndex = replayDist(generator);

		ReplaySample* pSample = pReplaySamples[replayIndex];

		for (int i = 0; i < _visible.size(); i++)
			_visible[i]._state = pSample->_visible[i];

		activate();

		float currentQ = value();

		updateOnError(gradientAlpha * (pSample->_q - currentQ));

		// If there is a next sample, we can update the action pointer
		if (replayIndex > 0) {
			int nextIndex = replayIndex - 1;

			// If q is same or improved, learn action to point
			if (pSample->_q > pSample->_originalQ) {

				// Activate
				for (int a = 0; a < _actions.size(); a++) {
					float sum = _actions[a]._bias._weight;

					for (int k = 0; k < _hidden.size(); k++)
						sum += _actions[a]._connections[k]._weight * _hidden[k]._state;

					_actions[a]._state = sum;
				}

				// Update
				for (int a = 0; a < _actions.size(); a++) {
					float alphaError = actionAlpha * (pReplaySamples[nextIndex]->_visible[a + _numState] - _actions[a]._state);

					_actions[a]._bias._weight += alphaError;

					for (int k = 0; k < _hidden.size(); k++)
						_actions[a]._connections[k]._weight += alphaError * _hidden[k]._state;
				}
			}
		}
	}

	_prevValue = predictedQ;

	for (int i = 0; i < _visible.size(); i++)
		_prevVisible[i] = _visible[i]._state;

	for (int i = 0; i < _hidden.size(); i++)
		_prevHidden[i] = _hidden[i]._state;
}

void FERL::activate() {
	for (int k = 0; k < _hidden.size(); k++) {
		float sum = _hidden[k]._bias._weight;

		for (int vi = 0; vi < _visible.size(); vi++)
			sum += _hidden[k]._connections[vi]._weight * _visible[vi]._state;

		_hidden[k]._state = sigmoid(sum);
	}
}

void FERL::updateOnError(float error) {
	// Update weights
	for (int k = 0; k < _hidden.size(); k++) {
		_hidden[k]._bias._weight += error * _hidden[k]._state;

		for (int vi = 0; vi < _visible.size(); vi++)
			_hidden[k]._connections[vi]._weight += error * _hidden[k]._state * _visible[vi]._state;
	}

	for (int vi = 0; vi < _visible.size(); vi++)
		_visible[vi]._bias._weight += error * _visible[vi]._state;
}

void FERL::saveToFile(std::ostream &os, bool saveReplayInformation) {
	os << _hidden.size() << " " << _visible.size() << " " << _numState << " " << _numAction << " " << _zInv << " " << _prevValue << std::endl;

	// Save hidden nodes
	for (int k = 0; k < _hidden.size(); k++) {
		os << _hidden[k]._state << " " << _hidden[k]._bias._weight;

		for (int vi = 0; vi < _visible.size(); vi++)
			os << " " << _hidden[k]._connections[vi]._weight;

		os << std::endl;
	}

	// Save visible nodes
	for (int vi = 0; vi < _visible.size(); vi++)
		os << _visible[vi]._state << " " << _visible[vi]._bias._weight << std::endl;

	if (saveReplayInformation) {
		os << "t" << _replaySamples.size() << std::endl;

		for (std::list<ReplaySample>::iterator it = _replaySamples.begin(); it != _replaySamples.end(); it++) {
			for (int i = 0; i < it->_visible.size(); i++)
				os << it->_visible[i] << " ";

			os << it->_q << std::endl;
		}
	}
	else
		os << "f" << std::endl;
}

void FERL::loadFromFile(std::istream &is, bool loadReplayInformation) {
	int numHidden, numVisible;

	is >> numHidden >> numVisible >> _numState >> _numAction >> _zInv >> _prevValue;

	_hidden.resize(numHidden);

	for (int k = 0; k < numHidden; k++) {
		_hidden[k]._connections.resize(numVisible);

		is >> _hidden[k]._state >> _hidden[k]._bias._weight;

		for (int vi = 0; vi < numVisible; vi++)
			is >> _hidden[k]._connections[vi]._weight;
	}

	_visible.resize(numVisible);

	for (int vi = 0; vi < numVisible; vi++)
		is >> _visible[vi]._state >> _visible[vi]._bias._weight;

	if (loadReplayInformation) {
		std::string hasReplayInformationString;

		is >> hasReplayInformationString;

		if (hasReplayInformationString == "t") {
			int numSamples;

			is >> numSamples;

			for (int i = 0; i < numSamples; i++) {
				ReplaySample rs;

				rs._visible.resize(numVisible);

				for (int j = 0; j < numVisible; j++)
					is >> rs._visible[j];

				is >> rs._q;

				_replaySamples.push_back(rs);
			}
		}
		else
			std::cerr << "Stream does not contain replay information, but the application tried to load it!" << std::endl;
	}
}
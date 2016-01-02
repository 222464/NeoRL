#pragma once

#include <vector>
#include <list>
#include <random>
#include <string>

namespace deep {
	class FERL {
	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		struct ReplaySample {
			std::vector<float> _visible;

			float _originalQ;
			float _q;
		};

	private:
		struct Connection {
			float _weight;
		};

		struct Hidden {
			Connection _bias;
			std::vector<Connection> _connections;
			float _state;

			Hidden()
				: _state(0.0f)
			{}
		};

		struct Visible {
			Connection _bias;
			float _state;

			Visible()
				: _state(0.0f)
			{}
		};

		std::vector<Hidden> _hidden;
		std::vector<Visible> _visible;
		std::vector<Hidden> _actions;

		int _numState;
		int _numAction;

		float _zInv;

		float _prevValue;

		std::vector<float> _prevVisible;
		std::vector<float> _prevHidden;

		std::list<ReplaySample> _replaySamples;

	public:
		FERL();

		void createRandom(int numState, int numAction, int numHidden, float weightStdDev, std::mt19937 &generator);

		void createFromParents(const FERL &parent1, const FERL &parent2, float averageChance, std::mt19937 &generator);

		void mutate(float perturbationStdDev, std::mt19937 &generator);

		// Returns action index
		void step(const std::vector<float> &state, std::vector<float> &action,
			float reward, float qAlpha, float gamma, float lambdaGamma,
			float actionAlpha, int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha,
			float breakChance, float perturbationStdDev,
			int maxNumReplaySamples, int replayIterations, float gradientAlpha,
			std::mt19937 &generator);

		void activate();
		void updateOnError(float error);

		float freeEnergy() const;

		void saveToFile(std::ostream &os, bool saveReplayInformation = false);
		void loadFromFile(std::istream &is, bool loadReplayInformation = false);

		float value() const {
			return -freeEnergy() * _zInv;
		}

		int getNumState() const {
			return _numState;
		}

		int getNumAction() const {
			return _numAction;
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		float getZInv() const {
			return _zInv;
		}

		const std::list<ReplaySample> &getSamples() const {
			return _replaySamples;
		}
	};
}
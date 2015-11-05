#pragma once

#include <vector>
#include <random>

namespace sdr {
	class IRSDR {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
		};

		struct HiddenNode {
			std::vector<Connection> _feedForwardConnections;
			std::vector<Connection> _recurrentConnections;

			float _state;
			float _statePrev;
			float _input;

			float _reconstruction;

			float _boost;

			HiddenNode()
				: _state(0.0f), _statePrev(0.0f), _reconstruction(0.0f), _input(0.0f), _boost(0.0f)
			{}
		};

		struct VisibleNode {
			float _input;
			float _reconstruction;

			VisibleNode()
				: _input(0.0f), _reconstruction(0.0f)
			{}
		};

	private:
		int _visibleWidth, _visibleHeight;
		int _hiddenWidth, _hiddenHeight;
		int _receptiveRadius;
		int _recurrentRadius;

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

		void pL(const std::vector<float> &states, float stepSize, float lambda, float hiddenDecay);

	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		void createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int recurrentRadius, float initMinWeight, float initMaxWeight, float initBoost, std::mt19937 &generator);

		void activate(int iter, float stepSize, float lambda, float hiddenDecay, float noise, std::mt19937 &generator);
		void reconstruct();
		void reconstruct(const std::vector<float> &states, std::vector<float> &reconHidden, std::vector<float> &reconVisible);
		void reconstructFeedForward(const std::vector<float> &states, std::vector<float> &recon);
		void learn(float learnFeedForward, float learnRecurrent, float learnBoost, float boostSparsity, float weightDecay, float maxWeightDelta = 0.1f);
		//void learn(const std::vector<float> &attentions, float learnFeedForward, float learnRecurrent);
		void stepEnd();

		void setVisibleState(int index, float value) {
			_visible[index]._input = value;
		}

		void setVisibleState(int x, int y, float value) {
			_visible[x + y * _visibleWidth]._input = value;
		}

		float getVisibleRecon(int index) const {
			return _visible[index]._reconstruction;
		}

		float getVisibleRecon(int x, int y) const {
			return _visible[x + y * _visibleWidth]._reconstruction;
		}

		float getVisibleState(int index) const {
			return _visible[index]._input;
		}

		float getVisibleState(int x, int y) const {
			return _visible[x + y * _visibleWidth]._input;
		}

		float getHiddenState(int index) const {
			return _hidden[index]._state;
		}

		float getHiddenState(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._state;
		}

		float getHiddenStatePrev(int index) const {
			return _hidden[index]._statePrev;
		}

		float getHiddenStatePrev(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._statePrev;
		}

		HiddenNode &getHiddenNode(int index) {
			return _hidden[index];
		}

		HiddenNode &getHiddenNode(int x, int y) {
			return _hidden[x + y * _hiddenWidth];
		}

		int getNumVisible() const {
			return _visible.size();
		}

		int getNumHidden() const {
			return _hidden.size();
		}

		int getVisibleWidth() const {
			return _visibleWidth;
		}

		int getVisibleHeight() const {
			return _visibleHeight;
		}

		int getHiddenWidth() const {
			return _hiddenWidth;
		}

		int getHiddenHeight() const {
			return _hiddenHeight;
		}

		int getReceptiveRadius() const {
			return _receptiveRadius;
		}

		float getVHWeight(int hi, int ci) const {
			return _hidden[hi]._feedForwardConnections[ci]._weight;
		}

		float getVHWeight(int hx, int hy, int ci) const {
			return _hidden[hx + hy * _hiddenWidth]._feedForwardConnections[ci]._weight;
		}

		void getVHWeights(int hx, int hy, std::vector<float> &rectangle) const;

		friend class HTSL;
	};
}
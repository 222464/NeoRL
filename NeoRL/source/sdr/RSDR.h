#pragma once

#include <vector>
#include <random>

namespace sdr {
	class RSDR {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
		};

		struct HiddenNode {
			std::vector<Connection> _feedForwardConnections;
			std::vector<unsigned short> _lateralConnections;
			std::vector<Connection> _recurrentConnections;

			float _threshold;

			float _state;
			float _statePrev;

			float _activation;

			float _reconstruction;

			HiddenNode()
				: _state(0.0f), _statePrev(0.0f), _activation(0.0f), _reconstruction(0.0f)
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
		int _inhibitionRadius;
		int _recurrentRadius;

		std::vector<VisibleNode> _visible;
		std::vector<HiddenNode> _hidden;

	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		void createRandom(int visibleWidth, int visibleHeight, int hiddenWidth, int hiddenHeight, int receptiveRadius, int inhibitionRadius, int recurrentRadius, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator);

		void activate(float sparsity);
		void inhibit(float sparsity, const std::vector<float> &activations, std::vector<float> &states);
		void reconstruct();
		void reconstructFeedForward(const std::vector<float> &states, std::vector<float> &recon);
		void learn(float learnFeedForward, float learnRecurrent, float learnLateral, float learnThreshold, float sparsity);
		void learn(const std::vector<float> &attentions, float learnFeedForward, float learnRecurrent, float learnLateral, float learnThreshold, float sparsity);
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

		float getHiddenActivation(int index) const {
			return _hidden[index]._activation;
		}

		float getHiddenActivation(int x, int y) const {
			return _hidden[x + y * _hiddenWidth]._activation;
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

		int getInhbitionRadius() const {
			return _inhibitionRadius;
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
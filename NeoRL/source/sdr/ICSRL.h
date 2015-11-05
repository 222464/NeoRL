#pragma once

#include "IPredictiveRSDR.h"
#include "../deep/SDRRL.h"

#include <algorithm>

namespace sdr {
	class ICSRL {
	public:
		struct ActionLayerDesc {
			int _cellCount;

			float _cellSparsity;

			ActionLayerDesc()
				: _cellCount(8), _cellSparsity(0.25f)
			{}
		};

		struct ActionNode {
			float _action;

			deep::SDRRL _sdrrl;

			std::vector<unsigned short> _feedBackConnectionIndices;
			std::vector<unsigned short> _lateralConnectionIndices;

			ActionNode()
				: _action(0.0f)
			{}
		};

		struct ActionLayer {
			std::vector<ActionNode> _actionNodes;
		};

	private:
		IPredictiveRSDR _prsdr;

		std::vector<ActionLayer> _actionLayers;

		std::vector<ActionLayerDesc> _actionLayerDescs;

		float _prevValue;

	public:
		static float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

		static float relu(float x, float leak) {
			return 1.0f + (x > 0.0f && x < 1.0f ? x : leak * x);
		}

		static float relud(float x, float leak) {
			return x > 0.0f && x < 1.0f ? 1.0f : leak;
		}

		float _qAlpha;
		float _actionAlpha;
		int _actionDeriveIterations;
		float _actionDeriveAlpha;
		float _reluLeak;

		float _explorationBreak;
		float _explorationStdDev;

		float _gamma;
		float _gammaLambda;

		ICSRL()
			: _qAlpha(0.1f),
			_actionAlpha(0.1f),
			_actionDeriveIterations(32),
			_actionDeriveAlpha(0.05f),
			_reluLeak(0.01f),
			_explorationBreak(0.01f),
			_explorationStdDev(0.05f),
			_gamma(0.99f),
			_gammaLambda(0.98f)
		{}

		void createRandom(int inputWidth, int inputHeight, int inputFeedBackRadius, const std::vector<ActionLayerDesc> &actionLayerDescs, const std::vector<IPredictiveRSDR::LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, float initBoost, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

		void setState(int index, float state) {
			_prsdr.setInput(index, state);
		}

		void setState(int x, int y, float state) {
			_prsdr.setInput(x, y, state);
		}

		float getAction(int index) const {
			return _actionLayers.front()._actionNodes[index]._action;
		}

		float getAction(int x, int y) const {
			return _actionLayers.front()._actionNodes[x + y * _prsdr.getLayers().front()._sdr.getVisibleWidth()]._action;
		}

		float getActionRel(int index) const {
			return _actionLayers.front()._actionNodes[index]._action;
		}
	};
}
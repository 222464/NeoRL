#pragma once

#include "PredictiveRSDR.h"

#include <algorithm>

namespace sdr {
	class QPRSDR {
	public:
		struct Connection {
			unsigned short _index;

			float _weight;
			float _trace;

			Connection()
				: _trace(0.0f)
			{}
		};

		struct QFunctionNode {
			float _state;
			float _error;

			Connection _bias;
			
			std::vector<Connection> _feedForwardConnections;

			QFunctionNode()
				: _state(0.0f), _error(0.0f)
			{}
		};

		struct QFunctionLayer {
			std::vector<QFunctionNode> _qFunctionNodes;

			std::vector<Connection> _qConnections;
		};

		struct ActionNode {
			float _predictedAction;
			float _deriveAction;
			float _exploratoryAction;

			float _error;

			int _inputIndex;

			ActionNode()
				: _predictedAction(0.0f),
				_deriveAction(0.0f),
				_exploratoryAction(0.0f),
				_error(0.0f)
			{}
		};

	private:
		PredictiveRSDR _prsdr;

		std::vector<QFunctionLayer> _qFunctionLayers;

		std::vector<ActionNode> _actionNodes;
		std::vector<int> _actionNodeIndices;

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

		QPRSDR()
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

		void createRandom(int inputWidth, int inputHeight, const std::vector<int> &actionIndices, const std::vector<PredictiveRSDR::LayerDesc> &layerDescs, float initMinWeight, float initMaxWeight, float initThreshold, std::mt19937 &generator);

		void simStep(float reward, std::mt19937 &generator, bool learn = true);

		void setState(int index, float state) {
			_prsdr.setInput(index, state);
		}

		void setState(int x, int y, float state) {
			_prsdr.setInput(x, y, state);
		}

		float getAction(int index) const {
			return _actionNodes[_actionNodeIndices[index]]._exploratoryAction;
		}

		float getAction(int x, int y) const {
			return _actionNodes[_actionNodeIndices[x + y * _prsdr.getLayers().front()._sdr.getVisibleWidth()]]._exploratoryAction;
		}

		float getActionRel(int index) const {
			return _actionNodes[index]._exploratoryAction;
		}
	};
}
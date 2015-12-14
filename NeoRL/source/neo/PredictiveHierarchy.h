#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"

namespace neo {
	class PredictiveHierarchy {
	public:
		struct LayerDesc {
			cl_int2 _size;

			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius;

			cl_float _scWeightAlpha;
			cl_float _scWeightRecurrentAlpha;
			cl_float _scWeightLambda;
			cl_float _scActiveRatio;
			cl_float _scBoostAlpha;

			cl_float _baseLineDecay;
			cl_float _baseLineSensitivity;

			cl_float _predWeightAlpha;

			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(4), _recurrentRadius(4), _lateralRadius(4), _feedBackRadius(4), _predictiveRadius(4),
				_scWeightAlpha(0.002f), _scWeightRecurrentAlpha(0.0001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.1f), _scBoostAlpha(0.02f),
				_baseLineDecay(0.01f), _baseLineSensitivity(0.01f),
				_predWeightAlpha(0.005f)
			{}
		};

		struct Layer {
			ComparisonSparseCoder _sc;
			Predictor _pred;

			DoubleBuffer2D _baseLines;

			cl::Image2D _reward;

			cl::Image2D _scHiddenStatesPrev;
		};

	private:
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

		cl::Kernel _baseLineUpdateKernel;
		cl::Kernel _baseLineUpdateSumErrorKernel;

	public:
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange, float initThreshold,
			std::mt19937 &rng);

		void simStep(sys::ComputeSystem &cs, const cl::Image2D &input, bool learn = true);

		void clearMemory(sys::ComputeSystem &cs);

		size_t getNumLayers() const {
			return _layers.size();
		}

		const Layer &getLayer(int index) const {
			return _layers[index];
		}

		const LayerDesc &getLayerDescs(int index) const {
			return _layerDescs[index];
		}

		const Predictor &getFirstLayerPred() const {
			return _layers.front()._pred;
		}
	};
}
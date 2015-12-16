#pragma once

#include "Helpers.h"

/*
ComparisonSparseCoder

Creates 2D sparse codes for a set of input layers.
*/
namespace neo {
	class ComparisonSparseCoder {
	public:
		struct VisibleLayerDesc {
			cl_int2 _size;

			cl_int _radius;

			cl_float _weightAlpha;
			cl_float _weightLambda;

			bool _ignoreMiddle;

			bool _useTraces;

			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(4), _weightAlpha(0.001f), _weightLambda(0.95f),
				_ignoreMiddle(false), _useTraces(false)
			{}
		};

		struct VisibleLayer {
			cl::Image2D _reconstructionError;

			DoubleBuffer3D _weights;

			cl_float2 _hiddenToVisible;
			cl_float2 _visibleToHidden;

			cl_int2 _reverseRadii;
		};

	private:
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenBiases;

		cl_int _lateralRadius;

		cl_int2 _hiddenSize;

		DoubleBuffer2D _hiddenActivationSummationTemp;
		DoubleBuffer2D _hiddenErrorSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

		cl::Kernel _forwardErrorKernel;
		cl::Kernel _activateKernel;
		cl::Kernel _activateIgnoreMiddleKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnHiddenBiasesKernel;
		cl::Kernel _learnHiddenWeightsKernel;
		cl::Kernel _learnHiddenWeightsTracesKernel;

		void reconstructError(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates);

	public:
		// Create with randomly initialized weights
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float initThreshold,
			std::mt19937 &rng);

		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio);

		void learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float boostAlpha, float activeRatio);
		void learn(sys::ComputeSystem &cs, const cl::Image2D &rewards, std::vector<cl::Image2D> &visibleStates, float boostAlpha, float activeRatio);

		void clearMemory(sys::ComputeSystem &cs);

		size_t getNumVisibleLayers() const {
			return _visibleLayers.size();
		}

		const VisibleLayer &getVisibleLayer(int index) const {
			return _visibleLayers[index];
		}

		const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		cl_int2 getHiddenSize() const {
			return _hiddenSize;
		}

		const DoubleBuffer2D &getHiddenStates() const {
			return _hiddenStates;
		}

		const DoubleBuffer2D &getHiddenActivations() const {
			return _hiddenActivationSummationTemp;
		}

		const DoubleBuffer2D &getHiddenBiases() const {
			return _hiddenBiases;
		}
	};
}
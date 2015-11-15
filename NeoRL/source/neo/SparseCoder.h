#pragma once

#include "Helpers.h"

/*
SparseCoder
	
Creates 2D sparse codes for a set of input layers.
*/
namespace neo {
	class SparseCoder {
	public:
		struct VisibleLayerDesc {
			cl_int2 _size;

			cl_int _radius;

			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(4)
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
		DoubleBuffer2D _hiddenSpikes;
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenActivations;
		DoubleBuffer2D _hiddenThresholds;

		DoubleBuffer3D _lateralWeights;

		cl_int _lateralRadius;

		cl_int2 _hiddenSize;

		DoubleBuffer2D _hiddenSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

		cl::Kernel _reconstructVisibleErrorKernel;
		cl::Kernel _activateFromReconstructionErrorKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnThresholdsKernel;
		cl::Kernel _learnWeightsKernel;
		cl::Kernel _learnWeightsTracesKernel;
		cl::Kernel _learnWeightsLateralKernel;

		void reconstructError(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates);

	public:
		// Create with randomly initialized weights
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
			cl_float2 initCodeRange, cl_float2 initReconstructionErrorRange,
			bool enableTraces,
			std::mt19937 &rng);

		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, cl_int settleIterations, cl_int measureIterations, cl_float leak);

		void learn(sys::ComputeSystem &cs, float weightAlpha, float weightLateralAlpha, float thresholdAlpha, float activeRatio);
		void learnTrace(sys::ComputeSystem &cs, const cl::Image2D &rewards, float weightAlpha, float weightLateralAlpha, float weightTraceLambda, float thresholdAlpha, float activeRatio);

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

		const DoubleBuffer2D &getHiddenThresholds() const {
			return _hiddenThresholds;
		}
	};
}
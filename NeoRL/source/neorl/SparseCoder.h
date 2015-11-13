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
				: _size({ 16, 16 }), _radius(4)
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
		DoubleBuffer2D _hiddenBoosts;

		cl_int2 _hiddenSize;

		DoubleBuffer2D _hiddenSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

		cl::Kernel _reconstructVisibleErrorKernel;
		cl::Kernel _activateFromReconstructionErrorKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnBoostsKernel;
		cl::Kernel _learnWeightsKernel;

		void reconstructError(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates);

	public:
		// Create with randomly initialized weights
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange, cl_float initBoost,
			std::mt19937 &rng);

		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, int iterations, float stepSize);

		void learn(sys::ComputeSystem &cs, float weightAlpha, float boostAlpha, float activeRatio);
	};
}
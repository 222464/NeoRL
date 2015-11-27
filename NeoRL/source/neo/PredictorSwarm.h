#pragma once

#include "Helpers.h"

/*
PredictorSwarm

Forms prediction mapping
*/
namespace neo {
	class PredictorSwarm {
	public:
		struct VisibleLayerDesc {
			cl_int2 _size;

			cl_int _radius;

			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(4)
			{}
		};

		struct VisibleLayer {
			DoubleBuffer3D _weights;

			cl_float2 _hiddenToVisible;
		};

	private:
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenActivations;

		cl_int2 _hiddenSize;

		DoubleBuffer2D _hiddenSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

		cl::Kernel _activateKernel;
		cl::Kernel _solveHiddenThresholdKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnWeightsTracesKernel;

	public:
		// Create with randomly initialized weights
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange,
			std::mt19937 &rng);

		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, bool threshold, float noise, std::mt19937 &rng);

		void learnTrace(sys::ComputeSystem &cs, float reward, float gamma, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, cl_float3 weightAlpha, cl_float2 weightLambda);

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
	};
}
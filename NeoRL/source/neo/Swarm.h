#pragma once

#include "Helpers.h"

/*
Swarm

Forms actions
*/
namespace neo {
	class Swarm {
	public:
		struct VisibleLayerDesc {
			cl_int2 _size;

			cl_int _qRadius;
			cl_int _startRadius;

			VisibleLayerDesc()
				: _size({ 8, 8 }), _qRadius(4), _startRadius(4)
			{}
		};

		struct VisibleLayer {
			cl::Image2D _predictedAction;

			cl::Image2D _actions;
			cl::Image2D _actionsExploratory;

			DoubleBuffer3D _qWeights;
			DoubleBuffer3D _startWeights;

			cl_float2 _hiddenToVisible;
			cl_float2 _visibleToHidden;

			cl_int2 _reverseQRadii;
		};

	private:
		DoubleBuffer2D _qStates;

		DoubleBuffer2D _hiddenStates;

		DoubleBuffer3D _qWeights;

		cl::Image2D _hiddenErrors;
		cl::Image2D _hiddenTD;

		cl_int2 _qSize;
		cl_int2 _hiddenSize;
		int _qRadius;

		cl_float2 _qToHidden;
		cl_float2 _hiddenToQ;

		cl_int2 _reverseQRadii;

		DoubleBuffer2D _hiddenSummationTemp;

		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;

		cl::Kernel _predictAction;
		cl::Kernel _qActivateToHiddenKernel;
		cl::Kernel _qActivateToQKernel;
		cl::Kernel _qSolveHiddenKernel;
		cl::Kernel _explorationKernel;
		cl::Kernel _qPropagateToHiddenErrorKernel;
		cl::Kernel _qPropagateToHiddenTDKernel;
		cl::Kernel _hiddenPropagateToVisibleActionKernel;
		cl::Kernel _startLearnWeightsKernel;
		cl::Kernel _qLearnVisibleWeightsTracesKernel;
		cl::Kernel _qLearnHiddenWeightsTracesKernel;

	public:
		// Create with randomly initialized weights
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 qSize, cl_int2 hiddenSize, int qRadius, cl_float2 initWeightRange,
			std::mt19937 &rng);

		void simStep(sys::ComputeSystem &cs, float reward, 
			const cl::Image2D &hiddenStatesFeedForward, const cl::Image2D &actionsFeedBack, 
			float expPert, float expBreak, int annealIterations, float actionAlpha, 
			float alphaHiddenQ, float alphaQ, float alphaPred, float lambda, std::mt19937 &rng);

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
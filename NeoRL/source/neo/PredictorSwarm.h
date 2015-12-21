#pragma once

#include "Helpers.h"

namespace neo {
	/*!
	\brief Swarm of predictors
	Can learn with OLPOMDP to perturb predictions
	*/
	class PredictorSwarm {
	public:
		/*!
		\brief Visible layer desc
		*/
		struct VisibleLayerDesc {
			/*!
			\brief Size of input layer
			*/
			cl_int2 _size;

			/*!
			\brief Radius onto input
			*/
			cl_int _radius;

			/*!
			\brief Initialize defaults
			*/
			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(4)
			{}
		};

		/*!
		\brief Visible layer
		*/
		struct VisibleLayer {
			/*!
			\brief Propagated prediction errors
			*/
			cl::Image2D _errors;

			/*!
			\brief Weights
			*/
			DoubleBuffer3D _weights;

			/*!
			\brief Transformations
			*/
			cl_float2 _hiddenToVisible;
			cl_float2 _visibleToHidden;
			//!@}

			/*!
			\brief Radius onto hidden (reverse from visible layer desc)
			*/
			cl_int2 _reverseRadii;
		};

	private:
		//!@{
		/*!
		\brief Hidden states and activations
		*/
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenActivations;
		//!@}

		/*!
		\brief Hidden size
		*/
		cl_int2 _hiddenSize;

		/*!
		\brief Summation temporary buffer
		*/
		DoubleBuffer2D _hiddenSummationTemp;

		//!@{
		/*!
		\brief Visible layers and descs
		*/
		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;
		//!@}

		//!@{
		/*!
		\brief Kernels
		*/
		cl::Kernel _activateKernel;
		cl::Kernel _solveHiddenThresholdKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnWeightsTracesKernel;
		cl::Kernel _errorPropagateKernel;
		//!@}

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange,
			std::mt19937 &rng);

		/*!
		\brief Activate predictor
		*/
		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, bool threshold, float noise, std::mt19937 &rng);

		/*!
		\brief Learn with RL + prediction error
		*/
		void learnTrace(sys::ComputeSystem &cs, float reward, float gamma, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, cl_float3 weightAlpha, cl_float2 weightLambda);

		/*!
		\brief Propagate prediction errors back to inputs based on targets
		*/
		void propagateError(sys::ComputeSystem &cs, const cl::Image2D &targets);

		/*!
		\brief Get number of visible layers
		*/
		size_t getNumVisibleLayers() const {
			return _visibleLayers.size();
		}

		/*!
		\brief Get access to a visible layer
		*/
		const VisibleLayer &getVisibleLayer(int index) const {
			return _visibleLayers[index];
		}

		/*!
		\brief Get access to a visible layer desc
		*/
		const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
			return _visibleLayerDescs[index];
		}

		/*!
		\brief Get hidden size
		*/
		cl_int2 getHiddenSize() const {
			return _hiddenSize;
		}

		/*!
		\brief Get hidden states
		*/
		const DoubleBuffer2D &getHiddenStates() const {
			return _hiddenStates;
		}
	};
}
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

		/*!
		\brief Inhibition temporary buffer
		*/
		cl::Image2D _inhibitionTemp;

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
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _solveHiddenThresholdKernel;
		cl::Kernel _inhibitKernel;
		cl::Kernel _learnWeightsTracesKernel;
		cl::Kernel _learnWeightsTracesInhibitedKernel;
		//!@}

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange,
			std::mt19937 &rng);

		//!@{
		/*!
		\brief Activate predictor
		Specify a non-one active ratio and non-negative-one inhibition radius to inhibit the result
		*/
		void activateInhibit(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float noise, float activeRatio, int inhibitionRadius, std::mt19937 &rng);
		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float noise, bool threshold, std::mt19937 &rng);
		//!@}

		//!@{
		/*!
		\brief Learn with RL + prediction error
		*/
		void learn(sys::ComputeSystem &cs, float reward, float gamma, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, cl_float2 weightAlpha, cl_float2 weightLambda, bool inhibited);
		//!@}

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
#pragma once

#include "Helpers.h"

namespace neo {
	/*!
	\brief Sparse coder
	Performs iterative sparse coding with explaining-away
	*/
	class SparseCoder {
	public:
		/*!
		\brief Visible layer desc
		*/
		struct VisibleLayerDesc {
			/*!
			\brief Size of layer
			*/
			cl_int2 _size;

			/*!
			\brief Radius
			*/
			cl_int _radius;

			//!@{
			/*!
			\brief Learning parameters
			*/
			cl_float _weightAlpha;
			cl_float _weightLambda;
			//!@}

			/*!
			\brief Whether or not to ignore middle node
			When recurrent, ignore middle node (self) if desired
			*/
			bool _ignoreMiddle;

			/*!
			\brief Whether or not to use eligibility traces
			*/
			bool _useTraces;

			/*!
			\brief Initialize defaults
			*/
			VisibleLayerDesc()
				: _size({ 8, 8 }), _radius(4), _weightAlpha(0.01f), _weightLambda(0.95f),
				_ignoreMiddle(false), _useTraces(false)
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

			//!@{
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
		\brief Spiking, resulting state, activations, and neuron threshold buffers
		*/
		DoubleBuffer2D _hiddenSpikes;
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenActivations;
		DoubleBuffer2D _hiddenThresholds;
		//!@}

		/*!
		\brief Lateral (inhibition) weights
		*/
		DoubleBuffer3D _lateralWeights;

		/*!
		\brief Lateral (inhibition) radius
		*/
		cl_int _lateralRadius;

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
		cl::Kernel _reconstructVisibleKernel;
		cl::Kernel _activateKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnThresholdsKernel;
		cl::Kernel _learnWeightsKernel;
		cl::Kernel _learnWeightsTracesKernel;
		cl::Kernel _learnWeightsLateralKernel;
		//!@}

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float2 initLateralWeightRange, cl_float initThreshold,
			std::mt19937 &rng);

		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, cl_int iterations, cl_float leak);

		//!@{
		/*!
		\brief Learn functions, with and without eligibility traces/rewards
		*/
		void learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float weightLateralAlpha, float thresholdAlpha, float activeRatio);
		void learn(sys::ComputeSystem &cs, const cl::Image2D &rewards, const std::vector<cl::Image2D> &visibleStates, float weightLateralAlpha, float thresholdAlpha, float activeRatio);
		//!@}

		/*!
		\brief Reconstruct (find input from sparse codes)
		*/
		void reconstruct(sys::ComputeSystem &cs, const cl::Image2D &hiddenStates, int visibleLayerIndex, cl::Image2D &visibleStates);

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
		\brief Get visible layer descs
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

		/*!
		\brief Get hidden thresholds
		*/
		const DoubleBuffer2D &getHiddenThresholds() const {
			return _hiddenThresholds;
		}
	};
}
#pragma once

#include "Helpers.h"

namespace neo {
	/*!
	\brief Comparison sparse coder
	Creates 2D sparse codes for a set of input layers
	*/
	class ComparisonSparseCoder {
	public:
		/*!
		\brief Desc for a visible layer
		*/
		struct VisibleLayerDesc {
			/*!
			\brief Size of visible layer
			*/
			cl_int2 _size;

			/*!
			\brief Radius onto the visible layer
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
			\brief Whether or not the center neuron (self in recurrent schemes) should be ignored
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
				: _size({ 8, 8 }), _radius(4), _weightAlpha(0.001f), _weightLambda(0.95f),
				_ignoreMiddle(false), _useTraces(false)
			{}
		};

		/*!
		\brief Visible layer
		*/
		struct VisibleLayer {
			/*!
			\brief Reconstruction error
			*/
			cl::Image2D _reconstructionError;

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
		\brief Hidden states and biases
		*/
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenBiases;
		//!@}

		/*!
		\brief Lateral (inhibition) radius
		*/
		cl_int _lateralRadius;

		/*!
		\brief Hidden size
		*/
		cl_int2 _hiddenSize;

		//!@{
		/*!
		\brief Temporary summation buffers
		*/
		DoubleBuffer2D _hiddenActivationSummationTemp;
		DoubleBuffer2D _hiddenErrorSummationTemp;
		//!@}

		//!@{
		/*!
		\brief Descs and layers
		*/
		std::vector<VisibleLayerDesc> _visibleLayerDescs;
		std::vector<VisibleLayer> _visibleLayers;
		//!@}

		//!@{
		/*!
		\brief Kernels
		*/
		cl::Kernel _forwardErrorKernel;
		cl::Kernel _activateKernel;
		cl::Kernel _activateIgnoreMiddleKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _learnHiddenBiasesKernel;
		cl::Kernel _learnHiddenWeightsKernel;
		cl::Kernel _learnHiddenWeightsTracesKernel;
		//!@}

		/*!
		\brief Reconstruct and find error with input for all visible layers
		*/
		void reconstructError(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates);

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs,
			cl_int2 hiddenSize, cl_int lateralRadius, cl_float2 initWeightRange, cl_float initThreshold,
			std::mt19937 &rng);

		/*!
		\brief Activate (find sparse codes)
		*/
		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio);

		//!@{
		/*!
		\brief Learn, with and without use of rewards + eligibility traces
		*/
		void learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float boostAlpha, float activeRatio);
		void learn(sys::ComputeSystem &cs, const cl::Image2D &rewards, std::vector<cl::Image2D> &visibleStates, float boostAlpha, float activeRatio);
		//!@}

		/*!
		\brief Clear working memory
		*/
		void clearMemory(sys::ComputeSystem &cs);

		/*!
		\brief Write to stream
		*/
		void writeToStream(sys::ComputeSystem &cs, std::ostream &os) const;

		/*!
		\brief Read from stream
		*/
		void readFromStream(sys::ComputeSystem &cs, sys::ComputeProgram &program, std::istream &is);

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

		/*!
		\brief Get hidden activations
		*/
		const DoubleBuffer2D &getHiddenActivations() const {
			return _hiddenActivationSummationTemp;
		}

		/*!
		\brief Get hidden biases
		*/
		const DoubleBuffer2D &getHiddenBiases() const {
			return _hiddenBiases;
		}
	};
}
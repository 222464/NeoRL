#pragma once

#include "Helpers.h"

namespace neo {
	/*!
	\brief Predictor
	Maps from 2D SDRs to a 2D prediction
	*/
	class Predictor {
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
		\brief Hidden states and activations
		*/
		DoubleBuffer2D _hiddenStates;
		//!@}

		/*!
		\brief Hidden size
		*/
		cl_int2 _hiddenSize;

		/*!
		\brief Hidden summation temprorary buffer
		*/
		DoubleBuffer2D _hiddenSummationTemp;

		//!@{
		/*!
		\brief Layers and descs
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
		cl::Kernel _learnWeightsKernel;
		cl::Kernel _learnWeightsTracesKernel;
		//!@}

		/*!
		\brief Remember if is using traces
		*/
		bool _useTraces;

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, cl_float2 initWeightRange,
			bool useTraces,
			std::mt19937 &rng);

		/*!
		\brief Activate predictor
		*/
		void activate(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, bool threshold = true);

		//!@{
		/*!
		\brief Learning functions
		*/
		void learn(sys::ComputeSystem &cs, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, float weightAlpha);
		void learn(sys::ComputeSystem &cs, float tdError, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, float weightAlpha, float weightLambda);
		void learnCurrent(sys::ComputeSystem &cs, const cl::Image2D &targets, std::vector<cl::Image2D> &visibleStatesPrev, float weightAlpha);
		//!@}

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
		\brief Get access to visible layer
		*/
		const VisibleLayer &getVisibleLayer(int index) const {
			return _visibleLayers[index];
		}

		/*!
		\brief Get access to visible layer
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
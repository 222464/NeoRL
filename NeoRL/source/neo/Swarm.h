#pragma once

#include "Helpers.h"

namespace neo {
	/*!
	\brief Swarm of overlapping SDRRL units
	Selects actions using deterministic policy gradients
	*/
	class Swarm {
	public:
		/*!
		\brief Visible layer desc
		*/
		struct VisibleLayerDesc {
			/*!
			\brief Size of layer
			*/
			cl_int2 _size;

			//!@{
			/*!
			\brief Radii
			*/
			cl_int _qRadius;
			cl_int _startRadius;
			//!@}

			/*!
			\brief Initialize defaults
			*/
			VisibleLayerDesc()
				: _size({ 8, 8 }), _qRadius(4), _startRadius(4)
			{}
		};

		/*!
		\brief Visibile layer
		*/
		struct VisibleLayer {
			/*!
			\brief Starting predicted action (non-exploratory)
			*/
			cl::Image2D _predictedAction;

			//!@{
			/*!
			\brief Actions, exporatory and non-exploratory
			*/
			cl::Image2D _actions;
			cl::Image2D _actionsExploratory;
			//!@}

			//!@{
			/*!
			\brief Weights for Q values and starting action prediction
			*/
			DoubleBuffer3D _qWeights;
			DoubleBuffer3D _startWeights;
			//!@}

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
			cl_int2 _reverseQRadii;
		};

	private:
		//!@{
		/*!
		\brief Q states, hidden states, and biases
		*/
		DoubleBuffer2D _qStates;
		DoubleBuffer2D _hiddenStates;
		DoubleBuffer2D _hiddenBiases;
		//!@}

		/*!
		\brief Q Weights
		*/
		DoubleBuffer3D _qWeights;

		//!@{
		/*!
		\brief Hidden errors and hidden temporal differences
		*/
		cl::Image2D _hiddenErrors;
		cl::Image2D _hiddenTD;
		//!@}

		//!@{
		/*!
		\brief Q and hidden sizes
		*/
		cl_int2 _qSize;
		cl_int2 _hiddenSize;
		//!@}

		/*!
		\brief Q radius
		*/
		int _qRadius;

		//!@{
		/*!
		\brief Q transformations
		*/
		cl_float2 _qToHidden;
		cl_float2 _hiddenToQ;
		//!@}

		/*!
		\brief Q radius onto Q layer (reverse from hidden layer)
		*/
		cl_int2 _reverseQRadii;

		/*!
		\brief Hidden summation temporary buffer
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
		cl::Kernel _predictAction;
		cl::Kernel _qInitSummationKernel;
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
		cl::Kernel _qLearnHiddenBiasesTracesKernel;
		//!@}

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 qSize, cl_int2 hiddenSize, int qRadius, cl_float2 initWeightRange,
			std::mt19937 &rng);

		/*!
		\brief Simulation stemp
		Requires the compute system, gating states, and RL parameters
		*/
		void simStep(sys::ComputeSystem &cs, float reward,
			const cl::Image2D &hiddenStatesFeedForward, const cl::Image2D &actionsFeedBack,
			float expPert, float expBreak, int annealIterations, float actionAlpha,
			float alphaHiddenQ, float alphaQ, float alphaPred, float lambda, float gamma, std::mt19937 &rng);

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
		\brief Get hidden layer size
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
		\brief Get hidden errors
		*/
		const cl::Image2D &getHiddenErrors() const {
			return _hiddenErrors;
		}

		/*!
		\brief Get hidden temporal differences
		*/
		const cl::Image2D &getHiddenTD() const {
			return _hiddenTD;
		}
	};
}
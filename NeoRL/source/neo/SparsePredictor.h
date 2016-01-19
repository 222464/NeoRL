#pragma once

#include "Helpers.h"

namespace neo {
	/*!
	\brief Sparse predictor
	Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
	*/
	class SparsePredictor {
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
			cl_int _encodeRadius;

			/*!
			\brief Radius onto hidden
			*/
			cl_int _predDecodeRadius;

			/*!
			\brief Radius onto higher prediction
			*/
			cl_int _feedBackDecodeRadius;

			/*!
			\brief Whether or not the predictions should be binary
			*/
			cl_int _predictBinary;

			/*!
			\brief Initialize defaults
			*/
			VisibleLayerDesc()
				: _size({ 8, 8 }), _encodeRadius(4), _predDecodeRadius(4), _feedBackDecodeRadius(4), _predictBinary(true)
			{}
		};

		/*!
		\brief Visible layer
		*/
		struct VisibleLayer {
			/*!
			\brief Predictions
			*/
			DoubleBuffer2D _predictions;

			/*!
			\brief Temporary error buffer
			*/
			cl::Image2D _error;

			//!@{
			/*!
			\brief Weights
			*/
			DoubleBuffer3D _encoderWeights; // Encoding weights (creates spatio-temporal sparse code)
			DoubleBuffer3D _predDecoderWeights; // Predictive decoding weights (points to t + 1)
			DoubleBuffer3D _feedBackDecoderWeights; // Feed back decoding weights (points to t + 1)
			//!@}

			//!@{
			/*!
			\brief Transformations
			*/
			cl_float2 _hiddenToVisible;
			cl_float2 _visibleToHidden;
			cl_float2 _visibleToFeedBack;
			//!@}
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
		\brief Feed back size
		*/
		std::vector<cl_int2> _feedBackSizes;

		/*!
		\brief Lateral (inhibitory) radius
		*/
		cl_int2 _lateralRadius;

		/*!
		\brief Hidden activation summation temporary buffer
		*/
		DoubleBuffer2D _hiddenActivationSummationTemp;

		/*!
		\brief Hidden error summation temporary buffer
		*/
		DoubleBuffer2D _hiddenErrorSummationTemp;

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
		cl::Kernel _encodeKernel;
		cl::Kernel _decodeKernel;
		cl::Kernel _solveHiddenKernel;
		cl::Kernel _predictionErrorKernel;
		cl::Kernel _errorPropagationKernel;
		cl::Kernel _learnEncoderWeightsKernel;
		cl::Kernel _learnDecoderWeightsKernel;
		//!@}

	public:
		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize, const std::vector<cl_int2> &feedBackSizes, cl_int lateralRadius, cl_float2 initWeightRange,
			std::mt19937 &rng);

		/*!
		\brief Activate predictor
		*/
		void activateEncoder(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio);
		void activateDecoder(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &feedBackStates);

		//!@{
		/*!
		\brief Learning functions
		*/
		void learn(sys::ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const std::vector<cl::Image2D> &visibleStatesPrev,
			const std::vector<cl::Image2D> &feedBackStatesPrev, const std::vector<cl::Image2D> &addidionalErrors, float weightAlpha, float weightLambda);
		//!@}

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
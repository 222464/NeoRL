#pragma once

#include "SparsePredictor.h"
#include "ImageWhitener.h"

namespace neo {
	/*!
	\brief Predictive hierarchy (no RL)
	*/
	class AgentPredQ {
	public:
		/*!
		\brief Layer desc
		*/
		struct LayerDesc {
			/*!
			\brief Size of layer
			*/
			cl_int2 _size;

			/*!
			\brief Radii
			*/
			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius;

			//!@{
			/*!
			\brief Sparse predictor parameters
			*/
			cl_float _spWeightEncodeAlpha;
			cl_float _spWeightDecodeAlpha;
			cl_float _spWeightLambda;
			cl_float _spActiveRatio;
			cl_float _spBiasAlpha;
			//!@}

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(5), _recurrentRadius(5), _lateralRadius(5), _feedBackRadius(6), _predictiveRadius(6),
				_spWeightEncodeAlpha(0.01f), _spWeightDecodeAlpha(0.01f), _spWeightLambda(0.95f),
				_spActiveRatio(0.04f), _spBiasAlpha(0.01f)
			{}
		};

		/*!
		\brief Layer
		*/
		struct Layer {
			/*!
			\brief Sparse predictor
			*/
			SparsePredictor _sp;

			/*!
			\brief Layer for additional error signals
			*/
			cl::Image2D _additionalErrors;
		};

	private:
		//!@{
		/*!
		\brief Store sizes
		*/
		cl_int2 _inputSize;
		cl_int2 _actionSize;
		cl_int2 _qSize;
		//!@}

		//!@{
		/*!
		\brief Layers and descs
		*/
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;

		cl::Image2D _qInputLayer;
		cl::Image2D _qRetrievalLayer;
		cl::Image2D _qTransforms;
		//!@}

		/*!
		\brief Input whitener
		*/
		ImageWhitener _inputWhitener;

		/*!
		\brief Zero layer for capping of the network
		*/
		cl::Image2D _zeroLayer;

		/*!
		\brief For RL
		*/
		float _prevValue;

		//!@{
		/*!
		\brief Additional kernels
		*/
		cl::Kernel _setQKernel;
		cl::Kernel _getQKernel;
		//!@}

	public:
		//!@{
		/*!
		\brief Whitening parameters
		*/
		cl_int _whiteningKernelRadius;
		cl_float _whiteningIntensity;
		//!@}

		//!@{
		/*!
		\brief For RL
		*/
		float _qAlpha;
		float _qGamma;
		//!@}

		/*!
		\brief Initialize defaults
		*/
		AgentPredQ()
			: _whiteningKernelRadius(1),
			_whiteningIntensity(1024.0f),
			// RL
			_prevValue(0.0f),
			_qAlpha(0.5f),
			_qGamma(0.98f)
		{}

		/*!
		\brief Create a predictive hierarchy with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int2 qSize, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange,
			std::mt19937 &rng);

		/*!
		\brief Simulation step of hierarchy
		*/
		void simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, const cl::Image2D &actionTaken, bool learn = true, bool whiten = false);

		/*!
		\brief Get number of layers
		*/
		size_t getNumLayers() const {
			return _layers.size();
		}

		/*!
		\brief Get access to a layer
		*/
		const Layer &getLayer(int index) const {
			return _layers[index];
		}

		/*!
		\brief Get access to a layer desc
		*/
		const LayerDesc &getLayerDescs(int index) const {
			return _layerDescs[index];
		}

		/*!
		\brief Get the prediction
		*/
		const cl::Image2D &getAction() const {
			return _layers.front()._sp.getVisibleLayer(2)._predictions[_back];
		}

		/*!
		\brief Get input whitener
		*/
		const ImageWhitener &getInputWhitener() const {
			return _inputWhitener;
		}
	};
}
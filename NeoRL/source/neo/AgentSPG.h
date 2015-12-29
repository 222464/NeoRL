#pragma once

#include "ComparisonSparseCoder.h"
#include "PredictorSwarm.h"

namespace neo {
	/*!
	\brief Predictive hierarchy (no RL)
	*/
	class AgentSPG {
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
			\brief Sparse coder parameters
			*/
			cl_float _scWeightAlpha;
			cl_float _scWeightRecurrentAlpha;
			cl_float _scWeightLambda;
			cl_float _scActiveRatio;
			cl_float _scBoostAlpha;
			//!@}

			/*!
			\brief Predictor parameters
			*/
			cl_float3 _predWeightAlpha;

			/*!
			\brief Exploration
			*/
			cl_float _noise;

			//!@{
			/*!
			\brief RL
			*/
			cl_float _gamma;
			cl_float2 _lambda;
			//!@}

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(5), _recurrentRadius(5), _lateralRadius(5), _feedBackRadius(6), _predictiveRadius(6),
				_scWeightAlpha(0.001f), _scWeightRecurrentAlpha(0.001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.06f), _scBoostAlpha(0.005f),
				_predWeightAlpha({ 1.0f, 0.002f, 0.1f }), _noise(0.05f),
				_gamma(0.99f), _lambda({ 0.95f, 0.95f })
			{}
		};

		/*!
		\brief Layer
		*/
		struct Layer {
			//!@{
			/*!
			\brief Sparse coder and predictor
			*/
			ComparisonSparseCoder _sc;
			PredictorSwarm _pred;
			//!@}

			/*!
			\brief Rewards for sparse coder
			*/
			cl::Image2D _reward;

			/*!
			\brief Previous hidden states
			*/
			cl::Image2D _scHiddenStatesPrev;
		};

	private:
		/*!
		\brief Store input size
		*/
		cl_int2 _inputSize;

		/*!
		\brief Store action size
		*/
		cl_int2 _actionSize;

		//!@{
		/*!
		\brief Layers and descs
		*/
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;
		//!@}

		//!@{
		/*!
		\brief Kernels for hierarchy
		*/
		cl::Kernel _predictionRewardKernel;
		//!@}

		/*!
		\brief Predictor (first layer)
		*/
		PredictorSwarm _firstLayerPred;

	public:
		/*!
		\brief Learning rate for first layer predictor
		*/
		cl_float3 _firstLayerPredWeightAlpha;

		/*!
		\brief Exploration for first layer predictor
		*/
		cl_float _firstLayerNoise;

		//!@{
		/*!
		\brief RL for first layer predictor
		*/
		cl_float _firstLayerGamma;
		cl_float2 _firstLayerLambda;
		//!@}

		/*!
		\brief Initialize defaults
		*/
		AgentSPG()
			: _firstLayerPredWeightAlpha({ 1.0f, 0.002f, 0.1f }),
			_firstLayerNoise(0.05f),
			_firstLayerGamma(0.99f), _firstLayerLambda({ 0.95f, 0.95f })
		{}

		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information.
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerFeedBackRadius, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange,
			std::mt19937 &rng);

		/*!
		\brief Simulation step of hierarchy
		*/
		void simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn = true);

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
		\brief Get first layer predictor (contains predictions for input)
		*/
		const PredictorSwarm &getFirstLayerPred() const {
			return _firstLayerPred;
		}
	};
}
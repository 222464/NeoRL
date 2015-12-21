#pragma once

#include "ComparisonSparseCoder.h"
#include "PredictorSwarm.h"

namespace neo {
	/*!
	\brief Stochastic policy gradient agent
	*/
	class AgentSPG {
	public:
		struct LayerDesc {
			/*!
			\brief Hidden size of the layer
			*/
			cl_int2 _hiddenSize;

			//!@{
			/*!
			\brief Radii
			*/
			cl_int _feedForwardRadius, _recurrentRadius, _lateralRadius, _feedBackRadius, _predictiveRadius;
			cl_int _qRadiusHiddenFeedForwardAttention, _qRadiusHiddenRecurrentAttention, _qRadiusHiddenAction, _qRadius;
			cl_int _startRadiusHiddenFeedForwardAttention, _startRadiusHiddenRecurrentAttention, _startRadiusHiddenAction;
			//!@}

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

			//!@{
			/*!
			\brief Baseline parameters
			*/
			cl_float _baseLineDecay;
			cl_float _baseLineSensitivity;
			//!@}

			//!@{
			/*!
			\brief Predictor parameters
			*/
			cl_float3 _predWeightAlpha;
			cl_float2 _predWeightLambda;
			//!@}

			/*!
			\brief Bellman equation parameters
			*/
			cl_float _gamma;

			/*!
			\brief Exploration parameters
			*/
			cl_float _noise;

			/*!
			\brief Attention parameters
			*/
			cl_float _minAttention;

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _hiddenSize({ 8, 8 }),
				_feedForwardRadius(4), _recurrentRadius(4), _lateralRadius(4), _feedBackRadius(6), _predictiveRadius(6),
				_qRadiusHiddenFeedForwardAttention(4), _qRadiusHiddenRecurrentAttention(4), _qRadiusHiddenAction(4), _qRadius(4),
				_startRadiusHiddenFeedForwardAttention(4), _startRadiusHiddenRecurrentAttention(4), _startRadiusHiddenAction(4),
				_scWeightAlpha(0.0002f), _scWeightRecurrentAlpha(0.00001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.05f), _scBoostAlpha(0.001f),
				_baseLineDecay(0.01f), _baseLineSensitivity(0.01f),
				_predWeightAlpha({ 0.005f, 0.002f, 0.01f }),
				_predWeightLambda({ 0.95f, 0.95f }),
				_gamma(0.99f), _noise(0.04f),
				_minAttention(0.05f)
			{}
		};

		/*!
		\brief Layer in hierarchy
		*/
		struct Layer {
			/*!
			\brief Sparse coder
			*/
			ComparisonSparseCoder _sc;

			//!@{
			/*!
			\brief Predictor swarms for action and attention
			*/
			PredictorSwarm _predAction;
			PredictorSwarm _predAttentionFeedForward;
			PredictorSwarm _predAttentionRecurrent;
			//!@}

			//!@{
			/*!
			\brief Temporary buffers
			*/
			cl::Image2D _modulatedFeedForwardInput;
			cl::Image2D _modulatedRecurrentInput;
			cl::Image2D _inhibitedAction;
			//!@}

			/*!
			\brief Baselines
			*/
			DoubleBuffer2D _baseLines;

			/*!
			\brief Sparse coder reward
			*/
			cl::Image2D _reward;

			/*!
			\brief Previous sparse coder hidden states
			*/
			cl::Image2D _scHiddenStatesPrev;
		};

	private:
		//!@{
		/*!
		\brief Layers and layer descs
		*/
		std::vector<Layer> _layers;
		std::vector<LayerDesc> _layerDescs;
		//!@}

		/*!
		\brief Output action
		*/
		cl::Image2D _action;

		//!@{
		/*!
		\brief Kernels
		*/
		cl::Kernel _baseLineUpdateKernel;
		cl::Kernel _baseLineUpdateSumErrorKernel;
		cl::Kernel _inhibitKernel;
		cl::Kernel _modulateKernel;
		cl::Kernel _copyActionKernel;
		//!@}

	public:
		/*!
		\brief Create an agent with random initialization
		Requires the compute system, program with the NeoRL kernels, input/action sizes, layer descs, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange, float initThreshold,
			std::mt19937 &rng);

		/*!
		\brief Simulation step of agent
		*/
		void simStep(sys::ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng);

		/*!
		\brief Clear working memory
		*/
		void clearMemory(sys::ComputeSystem &cs);

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
		\brief Get the action (exploratory)
		*/
		const cl::Image2D &getAction() const {
			return _action;
		}
	};
}
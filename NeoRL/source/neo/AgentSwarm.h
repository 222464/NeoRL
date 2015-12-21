#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"
#include "Swarm.h"

namespace neo {
	/*!
	\brief SDRRL swarm gradient agent
	*/
	class AgentSwarm {
	public:
		struct LayerDesc {
			/*!
			\brief Size of hidden layer in hierarchy
			*/
			cl_int2 _hiddenSize;

			/*!
			\brief Size of Q layer in hierarchy
			*/
			cl_int2 _qSize;

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

			/*!
			\brief Predictor parameters
			*/
			cl_float _predWeightAlpha;

			//!@{
			/*!
			\brief Swarm parameters
			*/
			cl_int _swarmAnnealingIterations;
			cl_float _swarmActionDeriveAlpha;
			cl_float _swarmQAlpha;
			cl_float _swarmQHiddenAlpha;
			cl_float _swarmPredAlpha;
			cl_float _swarmLambda;
			cl_float _swarmGamma;
			cl_float _swarmExpPert;
			cl_float _swarmExpBreak;
			//!@}

			/*!
			\brief Attention parameters
			*/
			cl_float _minAttention;

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _hiddenSize({ 8, 8 }), _qSize({ 4, 4 }),
				_feedForwardRadius(4), _recurrentRadius(4), _lateralRadius(4), _feedBackRadius(4), _predictiveRadius(4),
				_qRadiusHiddenFeedForwardAttention(4), _qRadiusHiddenRecurrentAttention(4), _qRadiusHiddenAction(4), _qRadius(4),
				_startRadiusHiddenFeedForwardAttention(4), _startRadiusHiddenRecurrentAttention(4), _startRadiusHiddenAction(4),
				_scWeightAlpha(0.01f), _scWeightRecurrentAlpha(0.001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.06f), _scBoostAlpha(0.01f),
				_baseLineDecay(0.01f), _baseLineSensitivity(0.01f),
				_predWeightAlpha(0.01f),
				_swarmAnnealingIterations(17), _swarmActionDeriveAlpha(0.08f),
				_swarmQAlpha(0.0005f), _swarmQHiddenAlpha(0.001f),
				_swarmPredAlpha(0.5f), _swarmLambda(0.95f), _swarmGamma(0.99f),
				_swarmExpPert(0.1f), _swarmExpBreak(0.03f),
				_minAttention(0.05f)
			{}
		};

		/*!
		\brief Layer (Hidden + Q) in hierarchy
		*/
		struct Layer {
			/*!
			\brief Sparse coder
			*/
			ComparisonSparseCoder _sc;

			/*!
			\brief Predictor
			*/
			Predictor _pred;

			/*!
			\brief Swarm
			*/
			Swarm _swarm;

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
			\brief Reward for sparse coder
			*/
			cl::Image2D _reward;

			/*!
			\brief Previous hidden states
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
		\brief Action for last layer (all zeros, dummy buffer)
		*/
		cl::Image2D _lastLayerAction;

		//!@{
		/*!
		\brief Kernels
		*/
		cl::Kernel _baseLineUpdateKernel;
		cl::Kernel _baseLineUpdateSumErrorKernel;
		cl::Kernel _inhibitKernel;
		cl::Kernel _modulateKernel;
		//!@}

	public:
		/*!
		\brief Create an agent with random initialization
		Requires the compute system, program with the NeoRL kernels, input/action sizes, layer descs, and initialization information
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int firstLayerPredictorRadius, const std::vector<LayerDesc> &layerDescs,
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
		\brief Number of layers in hierarchy
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
		\brief Get predictor from first layer
		*/
		const Predictor &getFirstLayerPred() const {
			return _layers.front()._pred;
		}

		/*!
		\brief Get exploratory actions image
		*/
		const cl::Image2D &getExploratoryActions() const {
			return _layers.front()._swarm.getVisibleLayer(2)._actionsExploratory;
		}
	};
}
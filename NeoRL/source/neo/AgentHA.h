#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"
#include "PredictorSwarm.h"

namespace neo {
	/*!
	\brief Predictive hierarchy (no RL)
	*/
	class AgentHA {
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

			//!@{
			/*!
			\brief Predictor parameters
			*/
			cl_float _predWeightAlpha;
			//!@}

			//!@{
			/*!
			\brief RL
			*/
			cl_float _qAlpha;
			cl_float _qBiasAlpha;
			cl_float _qLambda;
			cl_int _qRadius;
			//!@}

			/*!
			\brief Exploration parameter
			*/
			cl_float _noise;

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(5), _recurrentRadius(5), _lateralRadius(5), _feedBackRadius(6), _predictiveRadius(6),
				_scWeightAlpha(0.0004f), _scWeightRecurrentAlpha(0.0004f), _scWeightLambda(0.95f),
				_scActiveRatio(0.05f), _scBoostAlpha(0.01f),
				_predWeightAlpha(0.02f),
				_qAlpha(0.01f), _qBiasAlpha(0.01f), _qLambda(0.95f), _qRadius(6),
				_noise(0.05f)
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
			Predictor _pred;
			//!@}

			//!@{
			/*!
			\brief Q Hierarchy data
			*/
			DoubleBuffer3D _qWeights;
			DoubleBuffer2D _qBiases;
			DoubleBuffer2D _qStates;
			cl::Image2D _qErrors;
			//!@}

			/*!
			\brief Rewards for sparse coder
			*/
			cl::Image2D _reward;
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

		/*!
		\brief Predictor for actions
		*/
		Predictor _actionPred;

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

		cl::Kernel _qForwardKernel;
		cl::Kernel _qLastForwardKernel;
		cl::Kernel _qBackwardKernel;
		cl::Kernel _qLastBackwardKernel;
		cl::Kernel _qFirstBackwardKernel;
		cl::Kernel _qWeightUpdateKernel;
		cl::Kernel _qLastWeightUpdateKernel;
		cl::Kernel _qActionUpdateKernel;

		cl::Kernel _explorationKernel;
		//!@}

		//!@{
		/*!
		\brief Q Hierarchy data
		*/
		DoubleBuffer3D _qLastWeights;
		DoubleBuffer2D _qLastBiases;
		DoubleBuffer2D _qLastStates;

		cl::Image2D _qFirstErrors;

		cl_float _prevValue;
		//!@}

		//!@{
		/*!
		\brief Action buffers
		*/
		cl::Image2D _action;
		cl::Image2D _actionExploratory;
		//!@}

	public:
		//!@{
		/*!
		\brief Last RL
		*/
		cl_int2 _qLastSize;
		cl_float _qGamma;
		cl_float _qLastAlpha;
		cl_float _qLastBiasAlpha;
		cl_float _qLastLambda;
		cl_int _qLastRadius;
		//!@}

		//!@{
		/*!
		\brief General RL parameters
		*/
		cl_int _actionImprovementIterations;
		cl_float _actionImprovementAlpha;

		cl_float _expPert;
		cl_float _expBreak;
		//!@}

		/*!
		\brief Action predictor parameters
		*/
		cl_float _predActionWeightAlpha;

		/*!
		\brief Initialize defaults
		*/
		AgentHA()
			: _prevValue(0.0f),
			_qLastSize({ 16, 16 }), _qGamma(0.99f),
			_qLastAlpha(0.001f), _qLastBiasAlpha(0.001f), _qLastLambda(0.95f), _qLastRadius(8),
			_actionImprovementIterations(1), _actionImprovementAlpha(0.2f),
			_expPert(0.01f), _expBreak(0.01f),
			_predActionWeightAlpha(0.2f)
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
		\brief Get exploratory action
		*/
		const cl::Image2D &getExploratoryAction() const {
			return _actionExploratory;
		}
	};
}
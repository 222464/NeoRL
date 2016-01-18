#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"
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
			cl_float _predWeightLambda;
			//!@}

			/*!
			\brief Prediction reward parameters
			*/
			cl_float _predRewardBaselineDecay;

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(5), _recurrentRadius(5), _lateralRadius(5), _feedBackRadius(6), _predictiveRadius(6),
				_scWeightAlpha(0.001f), _scWeightRecurrentAlpha(0.001f), _scWeightLambda(0.95f),
				_scActiveRatio(0.05f), _scBoostAlpha(0.01f),
				_predWeightAlpha(0.01f), _predWeightLambda(0.95f), _predRewardBaselineDecay(0.01f)
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
			\brief For prediction reward determination
			*/
			DoubleBuffer2D _predRewardBaselines;
			cl::Image2D _predReward;
			cl::Image2D _propagatedPredReward;
			//!@}
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
		\brief Store Q size
		*/
		cl_int2 _qSize;

		/*!
		\brief Store input coder size
		*/
		cl_int2 _inputCoderSize;

		/*!
		\brief Store action coder size
		*/
		cl_int2 _actionCoderSize;

		/*!
		\brief Store Q coder size
		*/
		cl_int2 _qCoderSize;

		/*!
		\brief Q Input layer
		*/
		cl::Image2D _qInput;

		/*!
		\brief Transformation on Q values
		*/
		cl::Image2D _qTransform;

		/*!
		\brief Additional predictor for Q
		*/
		Predictor _qPred;

		//!@{
		/*!
		\brief Separate sparse coders for input to SDR transforms
		*/
		ComparisonSparseCoder _inputCoder;
		ComparisonSparseCoder _actionCoder;
		ComparisonSparseCoder _qCoder;
		//!@}

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
		cl::Kernel _predictionRewardPropagationKernel;
		cl::Kernel _setQKernel;
		//!@}

		/*!
		\brief Input whiteners
		*/
		ImageWhitener _inputWhitener;
		ImageWhitener _actionWhitener;
		ImageWhitener _qWhitener;

		//!@{
		/*!
		\brief Remember previous Q
		*/
		float _prevValue;
		float _prevQ;
		float _prevTDError;
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
		\brief RL parameters
		*/
		cl_float _qGamma;
		cl_float _qAlpha;
		cl_float _qWeightAlpha;
		cl_float _qWeightLambda;
		//!@}

		//!@{
		/*!
		\brief Coder parameters
		*/
		cl_int _inputCoderFeedForwardRadius;
		cl_int _actionCoderFeedForwardRadius;
		cl_int _qCoderFeedForwardRadius;

		cl_int _inputCoderLateralRadius;
		cl_int _actionCoderLateralRadius;
		cl_int _qCoderLateralRadius;

		cl_float _inputCoderActiveRatio;
		cl_float _actionCoderActiveRatio;
		cl_float _qCoderActiveRatio;

		cl_float _inputCoderAlpha;
		cl_float _actionCoderAlpha;
		cl_float _qCoderAlpha;

		cl_float _inputCoderBoostAlpha;
		cl_float _actionCoderBoostAlpha;
		cl_float _qCoderBoostAlpha;
		//!@}	

		/*!
		\brief Initialize defaults
		*/
		AgentPredQ()
			: _whiteningKernelRadius(1),
			_whiteningIntensity(1024.0f),
			_qGamma(0.98f), _qAlpha(0.5f),
			_qWeightAlpha(0.001f), _qWeightLambda(0.95f),
			_prevValue(0.0f), _prevQ(0.0f), _prevTDError(0.0f),
			_inputCoderFeedForwardRadius(4), _actionCoderFeedForwardRadius(4), _qCoderFeedForwardRadius(4),
			_inputCoderLateralRadius(4), _actionCoderLateralRadius(4), _qCoderLateralRadius(4),
			_inputCoderActiveRatio(0.05f), _actionCoderActiveRatio(0.05f), _qCoderActiveRatio(0.05f),
			_inputCoderAlpha(0.01f), _actionCoderAlpha(0.01f), _qCoderAlpha(0.01f),
			_inputCoderBoostAlpha(0.01f), _actionCoderBoostAlpha(0.01f), _qCoderBoostAlpha(0.01f)
		{}

		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information.
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int2 qSize, 
			cl_int2 inputCoderSize, cl_int2 actionCoderSize, cl_int2 qCoderSize,
			const std::vector<LayerDesc> &layerDescs,
			cl_float2 initWeightRange,
			std::mt19937 &rng);

		/*!
		\brief Simulation step of hierarchy
		*/
		void simStep(sys::ComputeSystem &cs, const cl::Image2D &input, const cl::Image2D &actionTaken, float reward, std::mt19937 &rng, bool learn = true, bool whiten = true);

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
		\brief Get the prediction
		*/
		const cl::Image2D &getAction() const {
			return _layers.front()._pred.getHiddenStates()[_back];
		}

		/*!
		\brief Get input whitener
		*/
		const ImageWhitener &getInputWhitener() const {
			return _inputWhitener;
		}

		/*!
		\brief Get aciton whitener
		*/
		const ImageWhitener &getActionWhitener() const {
			return _actionWhitener;
		}

		/*!
		\brief Get Q whitener
		*/
		const ImageWhitener &getQWhitener() const {
			return _qWhitener;
		}
	};
}
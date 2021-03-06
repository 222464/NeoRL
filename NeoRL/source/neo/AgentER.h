#pragma once

#include "ComparisonSparseCoder.h"
#include "Predictor.h"
#include "ImageWhitener.h"

#include <list>

namespace neo {
	/*!
	\brief Predictive hierarchy (no RL)
	*/
	class AgentER {
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
			cl_float _scActiveRatio;
			cl_float _scBoostAlpha;
			//!@}

			//!@{
			/*!
			\brief Predictor parameters
			*/
			cl_float _predWeightAlpha;
			//!@}

			/*!
			\brief Initialize defaults
			*/
			LayerDesc()
				: _size({ 8, 8 }),
				_feedForwardRadius(5), _recurrentRadius(0), _lateralRadius(5), _feedBackRadius(6), _predictiveRadius(6),
				_scWeightAlpha(0.001f), _scWeightRecurrentAlpha(0.001f),
				_scActiveRatio(0.05f), _scBoostAlpha(0.01f),
				_predWeightAlpha(0.01f)
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
			cl::Image2D _predReward;
			cl::Image2D _propagatedPredReward;
			//!@}

			//!@{
			/*!
			\brief For replay
			*/
			DoubleBuffer2D _scStatesTemp;
			DoubleBuffer2D _predStatesTemp;
			//!@}
		};

		/*!
		\brief Replay buffer frame
		*/
		struct ReplayFrame {
			std::vector<std::vector<int>> _layerStateBitIndices;
			std::vector<std::vector<int>> _layerPredBitIndices;

			std::vector<float> _prevExploratoryAction;
			std::vector<float> _prevBestAction;

			float _q;
			float _originalQ;
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
		\brief Q Input layer
		*/
		cl::Image2D _qInput;

		/*!
		\brief Q Target layer
		*/
		cl::Image2D _qTarget;

		/*!
		\brief Action target layer
		*/
		cl::Image2D _actionTarget;

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

		/*!
		\brief Experience replay buffer
		*/
		std::list<ReplayFrame> _frames;

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

		int _maxReplayFrames;
		int _replayIterations;
		//!@}

		/*!
		\brief Initialize defaults
		*/
		AgentER()
			: _whiteningKernelRadius(1),
			_whiteningIntensity(1024.0f),
			_qGamma(0.98f), _qAlpha(0.5f),
			_qWeightAlpha(0.01f),
			_maxReplayFrames(600), _replayIterations(10),
			_prevValue(0.0f), _prevQ(0.0f), _prevTDError(0.0f)
		{}

		/*!
		\brief Create a comparison sparse coder with random initialization
		Requires the compute system, program with the NeoRL kernels, and initialization information.
		*/
		void createRandom(sys::ComputeSystem &cs, sys::ComputeProgram &program,
			cl_int2 inputSize, cl_int2 actionSize, cl_int2 qSize,
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
#include "Settings.h"

#include "neo/PredictiveHierarchy.h"

#include <ctime>
#include <iostream>

#if EXPERIMENT_SELECTION == EXPERIMENT_SEQUENCE_RECALL

int main() {
	std::mt19937 generator(std::time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 4, 4);

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 12, 12 };
	layerDescs[2]._size = { 8, 8 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { 4, 4 }, 12, layerDescs, { -0.01f, 0.01f }, generator);

	std::uniform_int_distribution<int> item_dist(0, 9);

	std::vector<float> inputVec(16, 0.0f);

	float avg_error = 0.0f;

	for (int train_iter = 0; train_iter < 1000; train_iter++) {
		std::vector<int> items(10);

		for (int show_iter = 0; show_iter < 10; show_iter++) {
			items[show_iter] = item_dist(generator);

			for (int i = 0; i < 16; i++)
				inputVec[i] = 0.0f;

			inputVec[items[show_iter]] = 1.0f;

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, inputVec.data());

			ph.simStep(cs, inputImage);
		}

		for (int i = 0; i < 16; i++)
			inputVec[i] = 0.0f;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, inputVec.data());

		for (int wait_iter = 0; wait_iter < 10; wait_iter++) {
			ph.simStep(cs, inputImage);
		}

		// Show delimiter (item = 10)
		for (int i = 0; i < 16; i++)
			inputVec[i] = 0.0f;

		inputVec[10] = 1.0f;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, inputVec.data());

		ph.simStep(cs, inputImage);

		float error = 0.0f;

		std::vector<float> pred(16, 0.0f);

		for (int recall_iter = 0; recall_iter < 10; recall_iter++) {
			cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, pred.data());

			for (int i = 0; i < 16; i++) {
				if (i == items[recall_iter])
					error += std::pow(1.0f - pred[i], 2);
				else
					error += std::pow(0.0f - pred[i], 2);
			}

			for (int i = 0; i < 16; i++)
				inputVec[i] = 0.0f;

			inputVec[items[recall_iter]] = 1.0f;

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, inputVec.data());

			ph.simStep(cs, inputImage);
		}

		std::cout << error << std::endl;
	}

	return 0;
}

#endif
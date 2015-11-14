#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_SINUS_EXAMPLE

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/PredictiveHierarchy.h>

#include <time.h>
#include <iostream>
#include <random>

float sig(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_cpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 2, 2);

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { 2, 2 }, 8, layerDescs, { -0.01f, 0.01f }, 0.05f, { -0.01f, 0.01f }, { -0.01f, 0.01f }, generator);


	for (int i = 0; i < 1000; i++) {
		float v = std::sin(i * 0.1f);

		std::vector<float> vals(4);
		vals[0] = v;
		vals[1] = v - 1.0f;
		vals[2] = v + 1.0f;
		vals[3] = v * 2.0f;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 2, 2, 1 }, 0, 0, vals.data());

		ph.simStep(cs, inputImage);

		std::vector<float> res(4);

		cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 2, 2, 1 }, 0, 0, res.data());

		std::vector<float> sdr(256);

		cs.getQueue().enqueueReadImage(ph.getLayer(0)._sc.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 8, 8, 1 }, 0, 0, sdr.data());

		/*for (int x = 0; x < 8; x++) {
			for (int y = 0; y < 8; y++)
				std::cout << sdr[x + y * 8] << " ";

			std::cout << std::endl;
		}*/


		std::cout << res[0] << std::endl;
	}

	return 0;
}

#endif
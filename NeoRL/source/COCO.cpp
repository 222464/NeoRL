#include <Settings.h>

#if EXPERIMENT_SELECTION == EXPERIMENT_COCO

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vis/Plot.h>

#include <neo/PredictiveHierarchy.h>

#include <fstream>
#include <sstream>
#include <iostream>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 2, 2);

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 16, 16 };
	layerDescs[2]._size = { 16, 16 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { 2, 2 }, layerDescs, { -0.01f, 0.01f }, 0.0f, generator);
	//std::ifstream is("binh_save.neo");

	//ph.readFromStream(cs, prog, is);

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1200, 600), "NeoRL - COCO", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	return 0;
}

#endif
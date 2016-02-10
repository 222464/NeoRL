#include <Settings.h>

#if EXPERIMENT_SELECTION == EXPERIMENT_BINH_TEST

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vis/Plot.h>

#include <neo/PredictiveHierarchy.h>

#include <fstream>
#include <sstream>
#include <iostream>

int main()
{
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels2.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	std::vector<float> inputBuffer(5 * 5, 0.0f);
	std::vector<float> inputTransform(2 * (5 * 5 - 1), 0.0f);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (int i = 0; i < inputTransform.size(); i++)
		inputTransform[i] = dist01(generator) * 2.0f - 1.0f;

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 5, 5);

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 8, 8 };
	layerDescs[0]._predictiveRadius = 8;
	layerDescs[0]._feedBackRadius = 8;
	layerDescs[1]._size = { 8, 8 };
	layerDescs[2]._size = { 8, 8 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { 5, 5 }, layerDescs, { -0.2f, 0.2f }, generator);

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1200, 600), "Binh's Test", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	vis::Plot plot;
	plot._curves.resize(2);
	plot._curves[0]._shadow = 0.1f;	// input
	plot._curves[1]._shadow = 0.1f;	// predict
	
	sf::RenderTexture plotRT;
	plotRT.create(1200, 600, false);

	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	float minCurve = -5.0f;
	float maxCurve = 5.0f;

	plotRT.setActive();
	plotRT.clear(sf::Color::White);

	const int maxBufferSize = 50;

	bool quit = false;
	bool autoplay = false;

	bool sPrev = false;

	int index = -1;
	do {
		sf::Event event;

		while (renderWindow.pollEvent(event)) {
			switch (event.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			autoplay = false;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
			autoplay = true;

		sPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

		if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
			index++;

			float value = std::sin(0.164f * 3.141596f * index + 0.25f) + 0.7f * std::sin(0.12352f * 3.141596f * index * 1.5f + 0.2154f) + 0.5f * std::sin(0.0612f * 3.141596f * index * 3.0f - 0.2112f);

			inputBuffer[0] = value;
			
			for (int i = 1; i < inputBuffer.size(); i++) {
				inputBuffer[i] = value * inputTransform[2 * (i - 1) + 0] + inputTransform[2 * (i - 1) + 1];
			}

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 5, 5, 1 }, 0, 0, inputBuffer.data());

			ph.simStep(cs, inputImage, true, true);

			std::vector<float> res(inputBuffer.size());

			cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, { 0, 0, 0 }, { 5, 5, 1 }, 0, 0, res.data());

			float v = res[0];

			// Plot target data
			vis::Point p;
			p._position.x = index;
			p._position.y = value;
			p._color = sf::Color::Red;
			plot._curves[0]._points.push_back(p);

			// Plot predicted data
			vis::Point p1;
			p1._position.x = index;
			p1._position.y = v;
			p1._color = sf::Color::Blue;
			plot._curves[1]._points.push_back(p1);

			if (plot._curves[0]._points.size() > maxBufferSize) {
				plot._curves[0]._points.erase(plot._curves[0]._points.begin());

				int firstIndex = 0;

				for (std::vector<vis::Point>::iterator it = plot._curves[0]._points.begin(); it != plot._curves[0]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;

				plot._curves[1]._points.erase(plot._curves[1]._points.begin());

				firstIndex = 0;

				for (std::vector<vis::Point>::iterator it = plot._curves[1]._points.begin(); it != plot._curves[1]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;
			}

			renderWindow.clear();

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minCurve, maxCurve), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxCurve - minCurve) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);
	
			plotRT.display();

			sf::Sprite plotSprite;
			plotSprite.setTexture(plotRT.getTexture());

			renderWindow.draw(plotSprite);
			renderWindow.display();

			std::vector<float> data(64);

			cs.getQueue().enqueueReadImage(ph.getLayer(0)._sp.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 8, 8, 1 }, 0, 0, data.data());

			for (int x = 0; x < 8; x++) {

				for (int y = 0; y < 8; y++)
					std::cout << data[x + y * 8] << " ";

				std::cout << std::endl;
			}
		}
	} while (!quit);

	return 0;

}


#endif
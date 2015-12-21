#include <Settings.h>

#if EXPERIMENT_SELECTION == EXPERIMENT_BINH_TEST

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vis/Plot.h>

#include <neo/PredictiveHierarchy.h>

#include <fstream>
#include <sstream>
#include <iostream>

//#define _USE_ECG_DATA

int main()
{
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

	//ph.createRandom(cs, prog, { 2, 2 }, layerDescs, { -0.01f, 0.01f }, 0.0f, generator);
	std::ifstream is("binh_save.neo");

	ph.readFromStream(cs, prog, is);

#ifdef _USE_ECG_DATA
	std::ifstream input("e:/ecgsyn.dat");
	float x, y, z;

#endif

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1200, 600), "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	vis::Plot plot;
	plot._curves.resize(2);
	plot._curves[0]._shadow = 0.0;	// input
	plot._curves[1]._shadow = 0.0;	// predict

	sf::RenderTexture plotRT;
	plotRT.create(1200, 600, false);
	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");
	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	float minReward = -5.0f;
	float maxReward = +5.0f;
	plotRT.setActive();
	plotRT.clear(sf::Color::White);
	const int plotSampleTicks = 6;

	const int maxBufferSize = 200;
	bool quit = false;
	bool autoplay = false;
	float anomalyOffset = 0.f;
	float anomalyFreq = 1.f;
	float anomalyAmpl = 1.f;
	float anomalyPhase = 0.f;

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
				
		if (!sPrev && sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
			std::ofstream os("binh_save.neo");

			ph.writeToStream(cs, os);
		}

		sPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

		if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
		{
			++index;
#ifdef _USE_ECG_DATA
			if (!input.eof())
				input >> x >> y >> z;
			else
				autoplay = 0;
			// z - index of PQRSTU: P=1, Q=2, R= 3, S=4, T=5, U=6
			float value = y*4;	// without amplifying the amplitude, it is very difficult to make a sequence pattern QRST
#else
			float value = anomalyOffset + anomalyAmpl*std::sin(0.125f * 3.141596f * index * anomalyFreq + anomalyPhase) + 0.5f * std::sin(0.3f * 3.141596f * index * anomalyFreq + anomalyPhase);
#endif

			std::vector<float> vals(4);

			vals[0] = value;
			vals[1] = value - 1.0f;
			vals[2] = value + 1.0f;
			vals[3] = value * 2.0f;

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 2, 2, 1 }, 0, 0, vals.data());

			ph.simStep(cs, inputImage);

			std::vector<float> res(4);

			cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 2, 2, 1 }, 0, 0, res.data());

			std::vector<float> sdr(64);

			cs.getQueue().enqueueReadImage(ph.getLayer(0)._sc.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 8, 8, 1 }, 0, 0, sdr.data());

			for (int x = 0; x < 8; x++) {
				for (int y = 0; y < 8; y++)
					std::cout << sdr[x + y * 8] << " ";

				std::cout << std::endl;
			}

			// plot target data
			vis::Point p;
			p._position.x = index;
			p._position.y = value;
			p._color = sf::Color::Red;
			plot._curves[0]._points.push_back(p);

			// plot predicted data
			vis::Point p1;
			p1._position.x = index;
			p1._position.y = res[0];
			p1._color = sf::Color::Blue;
			plot._curves[1]._points.push_back(p1);

			if (plot._curves[0]._points.size() > maxBufferSize)
			{
				plot._curves[0]._points.erase(plot._curves[0]._points.begin());
				int firstIndex = 0;
				for (std::vector<vis::Point>::iterator it = plot._curves[0]._points.begin(); it != plot._curves[0]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;

				plot._curves[1]._points.erase(plot._curves[1]._points.begin());
				firstIndex = 1;
				for (std::vector<vis::Point>::iterator it = plot._curves[1]._points.begin(); it != plot._curves[1]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;
			}

			renderWindow.clear();

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);
			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[1]._points.size()), sf::Vector2f(minReward, maxReward), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[1]._points.size() / 10.0f, (maxReward - minReward) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

			plotRT.display();

			sf::Sprite plotSprite;
			plotSprite.setTexture(plotRT.getTexture());

			renderWindow.draw(plotSprite);
			renderWindow.display();
		}
	} while (!quit);

	return 0;

}


#endif
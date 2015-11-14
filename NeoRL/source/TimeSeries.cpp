#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_TIMESERIES

#include <sdr/IPredictiveRSDR.h>

#include <simtree/SDRST.h>

#include <vis/Plot.h>

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

int main() {
	std::mt19937 generator(time(nullptr));

	std::vector<float> timeSeries;

	std::ifstream fromFile("resources/pressures.txt");

	if (!fromFile.is_open()) {
		std::cerr << "Could not open pressures.txt!" << std::endl;

		return 1;
	}

	float minVal = 999.0f;
	float maxVal = -999.0f;

	while (fromFile.good() && !fromFile.eof()) {
		float value;

		fromFile >> value;

		timeSeries.push_back(value);

		minVal = std::min(minVal, value);

		maxVal = std::max(maxVal, value);
	}

	// Rescale
	for (int i = 0; i < timeSeries.size(); i++) {
		timeSeries[i] = (timeSeries[i] - minVal) / std::max(0.0001f, (maxVal - minVal));
	}

	int partition = timeSeries.size() * 0.5f;

	/*timeSeries.clear();

	timeSeries.resize(10);
	timeSeries[0] = { 0.0f, 1.0f, 0.0f };
	timeSeries[1] = { 0.0f, 0.0f, 0.0f };
	timeSeries[2] = { 1.0f, 1.0f, 0.0f };
	timeSeries[3] = { 0.0f, 0.0f, 1.0f };
	timeSeries[4] = { 0.0f, 1.0f, 0.0f };
	timeSeries[5] = { 0.0f, 0.0f, 1.0f };
	timeSeries[6] = { 0.0f, 0.0f, 0.0f };
	timeSeries[7] = { 0.0f, 0.0f, 0.0f };
	timeSeries[8] = { 0.0f, 1.0f, 0.0f };
	timeSeries[9] = { 0.0f, 1.0f, 1.0f };*/

	std::vector<sdr::IPredictiveRSDR::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 24;
	layerDescs[0]._height = 24;

	layerDescs[1]._width = 16;
	layerDescs[1]._height = 16;

	layerDescs[2]._width = 8;
	layerDescs[2]._height = 8;

	sdr::IPredictiveRSDR prsdr;

	prsdr.createRandom(1, 1, 16, layerDescs, -0.01f, 0.01f, 0.01f, 0.05f, 0.1f, generator);

	float avgError = 1.0f;

	float avgErrorDecay = 0.1f;

	vis::Plot plot;

	plot._curves.push_back(vis::Curve());

	plot._curves[0]._name = "Squared Error";

	
	for (int iter = 0; iter < 5; iter++) {
		for (int i = 0; i < partition; i++) {
			float error = prsdr.getPrediction(0) - timeSeries[i];

			float error2 = error * error;

			avgError = (1.0f - avgErrorDecay) * avgError + avgErrorDecay * error2;

			prsdr.setInput(0, timeSeries[i]);

			prsdr.simStep(generator);

			if (i % 10 == 0) {
				std::cout << "Iteration " << i << ": " << avgError << std::endl;
			}
		}
	}

	int step = 0;

	for (int i = 0; i < partition; i++) {
		float error = prsdr.getPrediction(0) - timeSeries[i];

		float error2 = error * error;

		avgError = (1.0f - avgErrorDecay) * avgError + avgErrorDecay * error2;

		prsdr.setInput(0, timeSeries[i]);

		prsdr.simStep(generator, false);

		if (step % 10 == 0) {
			std::cout << "Iteration " << i << ": " << avgError << std::endl;
		}

		vis::Point p;

		p._position.x = step;
		p._position.y = avgError;
		p._color = sf::Color::Red;

		plot._curves[0]._points.push_back(p);

		step++;
	}

	sf::RenderTexture rt;

	rt.create(1024, 1024);

	sf::Texture lineGradientTexture;

	lineGradientTexture.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;

	tickFont.loadFromFile("resources/arial.ttf");

	plot.draw(rt, lineGradientTexture, tickFont, 1.0f, sf::Vector2f(0.0f, step), sf::Vector2f(0.0f, 1.0f), sf::Vector2f(128.0f, 128.0f), sf::Vector2f(500.0f, 0.1f), 2.0f, 3.0f, 1.5f, 3.0f, 20.0f, 6);

	rt.display();

	rt.getTexture().copyToImage().saveToFile("plot.png");

	return 0;
}

#endif
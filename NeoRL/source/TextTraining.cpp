#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_TEXT_TRAINING

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/PredictiveHierarchy.h>

#include <time.h>
#include <iostream>
#include <random>
#include <fstream>

#include <unordered_map>
#include <unordered_set>

#include <assert.h>

std::ifstream::pos_type numCharsInFile(const std::string &fileName) {
	std::ifstream in(fileName, std::ifstream::ate);
	return in.tellg();
}

int main() {
	sf::RenderWindow window;

	sf::ContextSettings glContextSettings;
	glContextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(800, 600), "Link", sf::Style::Default, glContextSettings);

	//window.setFramerateLimit(60);
	//window.setVerticalSyncEnabled(true);

	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	unsigned long fileSize = numCharsInFile("corpus.txt");
	
	std::cout << "File size: " << fileSize << std::endl;

	unsigned long trainSize = std::round(0.97 * fileSize);
	unsigned long testSize = fileSize - trainSize;

	std::cout << "Train size and test size: " << trainSize << " " << testSize << std::endl;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	std::ifstream fromFile("corpus.txt");

	int minimum = 0;
	int maximum = 255;

	int numInputs = maximum - minimum + 1;

	int inputsRoot = std::ceil(std::sqrt(static_cast<float>(numInputs)));

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[0]._predictiveRadius = 12;
	layerDescs[0]._feedBackRadius = 12;
	layerDescs[0]._predWeightAlpha = 0.01f;

	layerDescs[1]._size = { 16, 16 };

	layerDescs[2]._size = { 16, 16 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { inputsRoot, inputsRoot }, layerDescs, { -0.01f, 0.01f }, 0.0f, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputsRoot, inputsRoot);

	std::vector<float> input(inputsRoot * inputsRoot, 0.0f);
	std::vector<float> pred(inputsRoot * inputsRoot, 0.0f);
	char predChar = 0;

	// ---------------------------- Game Loop -----------------------------

	sf::View view = window.getDefaultView();

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	sf::Font font;
	if (!font.loadFromFile("C:/Windows/Fonts/Arial.ttf"))
		return 1;

	sf::Text avgText;
	avgText.setColor(sf::Color::Red);
	avgText.setFont(font);
	avgText.setPosition(sf::Vector2f(100.0f, 100.0f));

	bool modeTest = false;

	std::normal_distribution<float> noiseDist(0.0f, 1.0f);

	unsigned long charPosition = 0;

	double logLikelihoodSum = 0.0;
	double testSamplesSummed = 0.0;

	double errorSum = 0.0;

	float runningAverage = 1.0f;
	float runningAverageDecay = 0.005f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (!modeTest && window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::Tilde)) {
			modeTest = true;

			fromFile.seekg(trainSize);

			std::cout << "Testing..." << std::endl;
		}

		// Read character
		char c;

		if (modeTest) {
			c = fromFile.get();

			charPosition++;

			if (fromFile.eof()) {
				//charPosition = trainSize;
				//fromFile.seekg(trainSize);

				//std::cout << "Final Log Likelihood: " << std::to_string(logLikelihoodSum / testSamplesSummed) << std::endl;
				std::cout << "Final error: " << errorSum / testSamplesSummed << std::endl;

				return 0;
			}
		}
		else {
			c = fromFile.get();

			charPosition++;

			if (charPosition >= trainSize) {
				charPosition = 0;
				fromFile.seekg(std::ios::beg);
			}
		}

		if (charPosition % 100 == 0) {
			window.clear();

			window.draw(avgText);

			const float scale = 4.0f;

			window.display();
		}

		if (!modeTest && charPosition % 50000 == 49999) {
			std::ofstream saveFile("neo_save1.neo");

			ph.writeToStream(cs, saveFile);
		}

		for (int i = 0; i < inputsRoot * inputsRoot; i++)
			input[i] = 0.0f;

		int index = c - minimum;

		input[index] = 1.0f;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputsRoot), static_cast<cl::size_type>(inputsRoot), 1 }, 0, 0, input.data());

		ph.simStep(cs, inputImage, !modeTest);

		cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputsRoot), static_cast<cl::size_type>(inputsRoot), 1 }, 0, 0, pred.data());

		int predIndex = 0;

		for (int i = 1; i < numInputs; i++)
			if (pred[i] > pred[predIndex])
				predIndex = i;

		predChar = predIndex + minimum;

		char nextChar = fromFile.peek();

		if (!modeTest)
			std::cout << predChar;

		/*float minPred = 999999.0f;

		for (int i = 0; i < numInputs; i++)
			minPred = std::min(pred[i], minPred);

		// Normalize
		float total = 0.0f;

		for (int i = 0; i < numInputs; i++) {
			total += std::min(1.0f, std::max(0.0f, pred[i]));
		}

		float logLikelihood = std::log(std::min(1.0f, std::max(0.0f, pred[nextChar - minimum])) / total);

		avgText.setString("Log likelihood: " + std::to_string(logLikelihood) + " (total: " + std::to_string(total) + ")");
		*/

		float error = nextChar == predChar ? 0.0f : 1.0f;

		runningAverage = (1.0f - runningAverageDecay) * runningAverage + runningAverageDecay * error;

		avgText.setString("Error Average: " + std::to_string(runningAverage));

		if (modeTest) {
			//logLikelihoodSum += logLikelihood;
			errorSum += error;

			testSamplesSummed++;
		}
	} while (!quit);

	return 0;
}

#endif
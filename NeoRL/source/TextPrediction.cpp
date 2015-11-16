#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_TEXT_PREDICTION

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

int main() {
	sf::RenderWindow window;

	sf::ContextSettings glContextSettings;
	glContextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(800, 600), "Link", sf::Style::Default, glContextSettings);

	//window.setFramerateLimit(60);
	//window.setVerticalSyncEnabled(true);

	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_cpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	std::ifstream fromFile("corpus.txt");

	fromFile.seekg(0, std::ios::end);
	size_t size = fromFile.tellg();
	std::string test(size, ' ');
	fromFile.seekg(0);
	fromFile.read(&test[0], size);

	int minimum = 255;
	int maximum = 0;

	std::unordered_set<char> characters;

	for (int i = 0; i < test.length(); i++) {
		minimum = std::min(static_cast<int>(test[i]), minimum);
		maximum = std::max(static_cast<int>(test[i]), maximum);

		if (characters.find(test[i]) == characters.end())
			characters.insert(test[i]);
	}

	std::vector<char> indexToChar;
	std::unordered_map<char, int> charToIndex;

	// Map characters to indices and vice versa
	for (std::unordered_set<char>::iterator it = characters.begin(); it != characters.end(); it++) {
		indexToChar.push_back(*it);
		charToIndex[*it] = indexToChar.size() - 1;
	}

	int numInputs = indexToChar.size();

	int inputsRoot = std::ceil(std::sqrt(static_cast<float>(numInputs)));

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(4);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[0]._feedForwardRadius = 6;
	
	layerDescs[1]._size = { 16, 16 };

	layerDescs[2]._size = { 16, 16 };

	layerDescs[3]._size = { 16, 16 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { inputsRoot, inputsRoot }, 4, layerDescs, { -0.01f, 0.01f }, { 0.01f, 0.05f }, 0.1f, { -0.01f, 0.01f }, { -0.01f, 0.01f }, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputsRoot, inputsRoot);

	std::vector<float> input(inputsRoot * inputsRoot, 0.0f);
	std::vector<float> pred(inputsRoot * inputsRoot, 0.0f);

	// ---------------------------- Game Loop -----------------------------

	sf::View view = window.getDefaultView();

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	int current = 0;

	float averageError = 0.0f;

	sf::Font font;
	if (!font.loadFromFile("C:/Windows/Fonts/Arial.ttf"))
		return 1;

	sf::Text avgText;
	avgText.setColor(sf::Color::Red);
	avgText.setFont(font);
	avgText.setPosition(sf::Vector2f(100.0f, 100.0f));

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

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		
		{
			window.clear();
			
			window.draw(avgText);

			const float scale = 4.0f;

			window.display();
		}

		for (int i = 0; i < inputsRoot * inputsRoot; i++)
			input[i] = 0.0f;

		int index = charToIndex[test[current]];

		input[index] = 1.0f;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputsRoot), static_cast<cl::size_type>(inputsRoot), 1 }, 0, 0, input.data());

		ph.simStep(cs, inputImage);

		cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputsRoot), static_cast<cl::size_type>(inputsRoot), 1 }, 0, 0, pred.data());

		int predIndex = 0;

		for (int i = 1; i < numInputs; i++)
			if (pred[i] > pred[predIndex])
				predIndex = i;

		char predChar = indexToChar[predIndex];

		std::cout << predChar;

		current = (current + 1) % test.length();

		float error = 1.0f;

		if (predChar == test[current])
			error = 0.0f;

		for (int i = 0; i < pred.size(); i++)
			std::cout << (pred[i] > 0.5f ? 1 : 0) << " ";

		//std::cout << predChar << " " << test[current] << std::endl;

		averageError = 0.99f * averageError + 0.01f * error;
		avgText.setString("Avg Err: " + std::to_string(averageError));

		if (current == 0)
			std::cout << "\n";
	} while (!quit);

	return 0;
}

#endif
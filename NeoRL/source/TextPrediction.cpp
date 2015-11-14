#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_TEXT_PREDICTION

#include "Settings.h"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <sdr/IPredictiveRSDR.h>

#include <simtree/SDRST.h>

#include <time.h>
#include <iostream>
#include <random>
#include <fstream>

#include <duktape/duktape.h>

#include <unordered_map>
#include <unordered_set>

#include <assert.h>

int main() {
	sf::RenderWindow window;

	sf::ContextSettings glContextSettings;
	glContextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(800, 600), "Link", sf::Style::Default, glContextSettings);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	std::mt19937 generator(time(nullptr));

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

	int inputsRoot = std::ceil(std::sqrt(static_cast<float>(numInputs))) + 1;

	std::vector<sdr::IPredictiveRSDR::LayerDesc> layerDescs(4);

	layerDescs[0]._width = 16;
	layerDescs[0]._height = 16;

	layerDescs[1]._width = 16;
	layerDescs[1]._height = 16;

	layerDescs[2]._width = 16;
	layerDescs[2]._height = 16;

	layerDescs[3]._width = 16;
	layerDescs[3]._height = 16;

	sdr::IPredictiveRSDR prsdr;

	prsdr.createRandom(inputsRoot, inputsRoot, 16, layerDescs, -0.01f, 0.01f, 0.01f, 0.04f, 0.05f, generator);

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

			sf::Image sdr;

			sdr.create(prsdr.getLayerDescs().front()._width, prsdr.getLayerDescs().front()._height);

			for (int x = 0; x < sdr.getSize().x; x++)
				for (int y = 0; y < sdr.getSize().y; y++) {
					sf::Color c = sf::Color::White;

					c.r = c.g = c.b = prsdr.getLayers().front()._sdr.getHiddenState(x, y) * 255.0f;

					sdr.setPixel(x, y, c);
				}

			sf::Texture sdrTex;

			sdrTex.loadFromImage(sdr);

			sf::Sprite sdrS;

			sdrS.setTexture(sdrTex);

			sdrS.setPosition(0.0f, window.getSize().y - sdrTex.getSize().y * scale);

			sdrS.setScale(scale, scale);

			window.draw(sdrS);

			window.display();
		}

		for (int i = 0; i < inputsRoot * inputsRoot; i++)
			prsdr.setInput(i, 0.0f);

		int index = charToIndex[test[current]];

		prsdr.setInput(index, 1.0f);

		prsdr.simStep(generator);

		int predIndex = 0;

		for (int i = 1; i < numInputs; i++)
			if (prsdr.getPrediction(i) > prsdr.getPrediction(predIndex))
				predIndex = i;

		char predChar = indexToChar[std::min(numInputs - 1, predIndex)];

		std::cout << predChar;

		current = (current + 1) % test.length();

		float error = 1.0f;

		if (predChar == test[current])
			error = 0.0f;

		//std::cout << predChar << " " << test[current] << std::endl;

		averageError = 0.99f * averageError + 0.01f * error;
		avgText.setString("Avg Err: " + std::to_string(averageError));

		if (current == 0)
			std::cout << "\n";
	} while (!quit);

	return 0;
}

#endif
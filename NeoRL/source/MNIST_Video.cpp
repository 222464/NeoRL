#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_MNIST_VIDEO

#include <neo/PredictiveHierarchy.h>

#include <vis/Plot.h>

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

struct Image {
	std::vector<sf::Uint8> _intensities;
};


void loadMNISTimage(std::ifstream &fromFile, int index, Image &img) {
	const int headerSize = 16;
	const int imageSize = 28 * 28;

	fromFile.seekg(headerSize + index * imageSize);

	if (img._intensities.size() != 28 * 28)
		img._intensities.resize(28 * 28);

	fromFile.read(reinterpret_cast<char*>(img._intensities.data()), 28 * 28);
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 64, 64);

	std::ifstream fromFile("resources/train-images.idx3-ubyte", std::ios::binary | std::ios::in);

	if (!fromFile.is_open()) {
		std::cerr << "Could not open train-images.idx3-ubyte!" << std::endl;

		return 1;
	}

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(4);

	layerDescs[0]._size = { 64, 64 };
	layerDescs[0]._feedForwardRadius = 10;

	layerDescs[1]._size = { 48, 48 };

	layerDescs[2]._size = { 32, 32 };

	layerDescs[3]._size = { 24, 24 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { 64, 64 }, 8, layerDescs, { -0.01f, 0.01f }, { 0.01f, 0.05f }, 0.1f, { -0.01f, 0.01f }, { -0.01f, 0.01f }, generator);

	float avgError = 1.0f;

	float avgErrorDecay = 0.1f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1024, 512), "MNIST Video Test");

	vis::Plot plot;

	plot._curves.push_back(vis::Curve());

	plot._curves[0]._name = "Squared Error";

	std::uniform_int_distribution<int> digitDist(0, 59999);
	std::uniform_real<float> dist01(0.0f, 1.0f);

	sf::RenderTexture rt;

	rt.create(64, 64);

	sf::Image digit0;
	sf::Texture digit0Tex;
	sf::Image digit1;
	sf::Texture digit1Tex;

	sf::Image pred;
	sf::Texture predTex;

	digit0.create(28, 28);
	digit1.create(28, 28);

	pred.create(rt.getSize().x, rt.getSize().y);

	const float boundingSize = (64 - 28) / 2;
	const float center = 32;
	const float minimum = center - boundingSize;
	const float maximum = center + boundingSize;

	float avgError2 = 1.0f;

	const float avgError2Decay = 0.01f;

	std::vector<float> prediction(64 * 64, 0.0f);

	for (int iter = 0; iter < 10000; iter++) {
		// Select digit indices
		int d0 = digitDist(generator);
		int d1 = digitDist(generator);

		// Load digits
		Image img0, img1;

		loadMNISTimage(fromFile, d0, img0);
		loadMNISTimage(fromFile, d1, img1);

		for (int x = 0; x < digit0.getSize().x; x++)
			for (int y = 0; y < digit0.getSize().y; y++) {
				int index = x + y * digit0.getSize().x;

				sf::Color c = sf::Color::White;

				c.a = img0._intensities[index];

				digit0.setPixel(x, y, c);
			}

		digit0Tex.loadFromImage(digit0);

		for (int x = 0; x < digit1.getSize().x; x++)
			for (int y = 0; y < digit1.getSize().y; y++) {
				int index = x + y * digit1.getSize().x;

				sf::Color c = sf::Color::White;

				c.a = img1._intensities[index];

				digit1.setPixel(x, y, c);
			}

		digit1Tex.loadFromImage(digit1);

		sf::Vector2f vel0(dist01(generator) * 2.0f - 1.0f, dist01(generator) * 2.0f - 1.0f);
		sf::Vector2f vel1(dist01(generator) * 2.0f - 1.0f, dist01(generator) * 2.0f - 1.0f);

		sf::Vector2f pos0(dist01(generator) * (maximum - minimum) + minimum, dist01(generator) * (maximum - minimum) + minimum);
		sf::Vector2f pos1(dist01(generator) * (maximum - minimum) + minimum, dist01(generator) * (maximum - minimum) + minimum);

		float vel0mul = dist01(generator) * 6.0f / std::max(1.0f, std::sqrt(vel0.x * vel0.x + vel0.y + vel0.y));

		vel0 *= vel0mul;

		float vel1mul = dist01(generator) * 6.0f / std::max(1.0f, std::sqrt(vel1.x * vel1.x + vel1.y + vel1.y));

		vel1 *= vel1mul;

		// Render video
		for (int f = 0; f < 20; f++) {
			sf::Event windowEvent;

			while (window.pollEvent(windowEvent)) {
				switch (windowEvent.type) {
				case sf::Event::Closed:
					return 0;
				}
			}

			pos0 += vel0;

			pos1 += vel1;

			if (pos0.x < minimum) {
				pos0.x = minimum;

				vel0.x *= -1.0f;
			}
			else if (pos0.x > maximum) {
				pos0.x = maximum;

				vel0.x *= -1.0f;
			}

			if (pos0.y < minimum) {
				pos0.y = minimum;

				vel0.y *= -1.0f;
			}
			else if (pos0.y > maximum) {
				pos0.y = maximum;

				vel0.y *= -1.0f;
			}

			if (pos1.x < minimum) {
				pos1.x = minimum;

				vel1.x *= -1.0f;
			}
			else if (pos1.x > maximum) {
				pos1.x = maximum;

				vel1.x *= -1.0f;
			}

			if (pos1.y < minimum) {
				pos1.y = minimum;

				vel1.y *= -1.0f;
			}
			else if (pos1.y > maximum) {
				pos1.y = maximum;

				vel1.y *= -1.0f;
			}

			window.clear();
			rt.clear(sf::Color::Black);

			sf::Sprite s0;

			s0.setTexture(digit0Tex);

			s0.setOrigin(28 / 2, 28 / 2);

			s0.setPosition(pos0);

			rt.draw(s0);

			sf::Sprite s1;

			s1.setTexture(digit1Tex);

			s1.setOrigin(28 / 2, 28 / 2);

			s1.setPosition(pos1);

			rt.draw(s1);

			rt.display();

			// Get input image
			sf::Image res = rt.getTexture().copyToImage();

			// Show RT
			const float scale = 4.0f;

			sf::Sprite s;

			s.setScale(scale, scale);

			s.setTexture(rt.getTexture());

			window.draw(s);

			std::vector<float> input(64 * 64);

			// Train
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
				for (int x = 0; x < res.getSize().x; x++)
					for (int y = 0; y < res.getSize().y; y++) {
						input[x + y * 64] = prediction[x + y * 64];
					}
			}
			else {
				const float predictionIncorporateRatio = 0.1f;

				for (int x = 0; x < res.getSize().x; x++)
					for (int y = 0; y < res.getSize().y; y++) {
						input[x + y * 64] = (1.0f - predictionIncorporateRatio) * res.getPixel(x, y).r / 255.0f + predictionIncorporateRatio * prediction[x + y * 64];
					}
			}

			// Error
			float error = 0.0f;

			for (int x = 0; x < res.getSize().x; x++)
				for (int y = 0; y < res.getSize().y; y++) {
					error += std::pow(res.getPixel(x, y).r / 255.0f - prediction[x + y * 64], 2);
				}

			error /= res.getSize().x * res.getSize().y;

			avgError2 = (1.0f - avgError2Decay) * avgError2 + avgError2Decay * error;

			std::cout << "Squared Error: " << avgError2 << std::endl;

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 64, 64, 1 }, 0, 0, input.data());

			ph.simStep(cs, inputImage);

			cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 64, 64, 1 }, 0, 0, prediction.data());

			// Show prediction
			for (int x = 0; x < rt.getSize().x; x++)
				for (int y = 0; y < rt.getSize().y; y++) {
					sf::Color c = sf::Color::White;

					c.r = c.b = c.g = std::min(1.0f, std::max(0.0f, prediction[x + y * 64])) * 255.0f;

					pred.setPixel(x, y, c);
				}

			predTex.loadFromImage(pred);

			sf::Sprite sp;

			sp.setTexture(predTex);

			sp.setScale(scale, scale);

			sp.setPosition(window.getSize().x - scale * rt.getSize().x, 0);

			window.draw(sp);

			/*sf::Image sdr;

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

			window.draw(sdrS);*/

			window.display();
	
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
				return 0;
		}
	}

	/*sf::RenderTexture rt;

	rt.create(1024, 1024);

	sf::Texture lineGradientTexture;

	lineGradientTexture.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;

	tickFont.loadFromFile("resources/arial.ttf");

	plot.draw(rt, lineGradientTexture, tickFont, 1.0f, sf::Vector2f(0.0f, step), sf::Vector2f(0.0f, 1.0f), sf::Vector2f(128.0f, 128.0f), sf::Vector2f(500.0f, 0.1f), 2.0f, 3.0f, 1.5f, 3.0f, 20.0f, 6);

	rt.display();

	rt.getTexture().copyToImage().saveToFile("plot.png");*/

	return 0;
}

#endif
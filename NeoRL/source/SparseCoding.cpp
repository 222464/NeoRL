#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_SPARSE_CODING

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neorl/SparseCoder.h>

#include <time.h>
#include <iostream>
#include <random>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);
	
	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	const int sampleWidth = 32;
	const int sampleHeight = 32;
	const int codeWidth = 16;
	const int codeHeight = 16;
	const int stepsPerFrame = 1;

	// --------------------------- Create the Sparse Coder ---------------------------

	neo::SparseCoder sparseCoder;

	std::vector<neo::SparseCoder::VisibleLayerDesc> layerDescs(1);

	layerDescs[0]._size = { sampleWidth, sampleHeight };

	sparseCoder.createRandom(cs, prog, layerDescs, { codeWidth, codeHeight }, { -0.01f, 0.01f }, 0.1f, { -0.01f, 0.01f }, { -0.01f, 0.01f }, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), sampleWidth, sampleHeight);

	// ------------------------------- Load Resources --------------------------------

	sf::Image sampleImage;

	sampleImage.loadFromFile("testImage.png");

	sf::Texture sampleTexture;

	sampleTexture.loadFromImage(sampleImage);

	sf::Image reconstructionImage;
	reconstructionImage.loadFromFile("resources/noreconstruction.png");

	sf::Texture reconstructionTexture;
	reconstructionTexture.loadFromImage(reconstructionImage);

	// ------------------------------- Simulation Loop -------------------------------

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(1280, 720), "Sparse Coding", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);
	renderWindow.setFramerateLimit(60);

	std::uniform_int_distribution<int> widthDist(0, static_cast<int>(sampleImage.getSize().x) - sampleWidth - 1);
	std::uniform_int_distribution<int> heightDist(0, static_cast<int>(sampleImage.getSize().y) - sampleHeight - 1);

	std::normal_distribution<float> noiseDist(0.0f, 0.01f);

	float dt = 0.017f;

	bool quit = false;

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

		renderWindow.clear();

		// -------------------------- Sparse Coding ----------------------------

		for (int s = 0; s < stepsPerFrame; s++) {
			int sampleX = widthDist(generator);
			int sampleY = heightDist(generator);

			std::vector<float> inputf(sampleWidth * sampleHeight);

			for (int x = 0; x < sampleWidth; x++)
				for (int y = 0; y < sampleHeight; y++) {
					int tx = sampleX + x;
					int ty = sampleY + y;

					inputf[x + y * sampleWidth] = sampleImage.getPixel(tx, ty).r / 255.0f + noiseDist(generator);
				}

			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sampleWidth, sampleHeight, 1 };

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, origin, region, 0, 0, inputf.data());

			sparseCoder.activate(cs, std::vector<cl::Image2D>(1, inputImage), 30, 0.1f);

			sparseCoder.learn(cs, 0.01f, 0.01f, 0.1f);
		}

		/*if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
			// Perform reconstruction
			std::vector<float> recon(sampleImage.getSize().x * sampleImage.getSize().y, 0.0f);
			std::vector<float> sums(sampleImage.getSize().x * sampleImage.getSize().y, 0.0f);

			for (int wx = 0; wx < sampleImage.getSize().x - sampleWidth; wx += 4)
				for (int wy = 0; wy < sampleImage.getSize().y - sampleHeight; wy += 4) {
					std::vector<float> inputf(sampleWidth * sampleHeight);

					for (int x = 0; x < sampleWidth; x++)
						for (int y = 0; y < sampleHeight; y++) {
							int tx = wx + x;
							int ty = wy + y;

							inputf[x + y * sampleWidth] = sampleImage.getPixel(tx, ty).r / 255.0f;
						}

					for (int i = 0; i < inputf.size(); i++)
						sparseCoder.setVisibleState(i, inputf[i]);

					sparseCoder.activate(30, 5, 0.05f, 0.05f, generator);

					sparseCoder.stepEnd();

					for (int x = 0; x < sampleWidth; x++)
						for (int y = 0; y < sampleHeight; y++) {
							int tx = wx + x;
							int ty = wy + y;

							recon[tx + ty * sampleImage.getSize().x] += sparseCoder.getVisibleRecon(x, y);
							sums[tx + ty * sampleImage.getSize().x] += 1.0f;
						}
				}

			float minimum = 99999.0f;
			float maximum = -99999.0f;

			for (int x = 0; x < sampleImage.getSize().x; x++)
				for (int y = 0; y < sampleImage.getSize().y; y++) {
					recon[x + y * sampleImage.getSize().x] /= sums[x + y * sampleImage.getSize().x];

					minimum = std::min(minimum, recon[x + y * sampleImage.getSize().x]);
					maximum = std::max(maximum, recon[x + y * sampleImage.getSize().x]);
				}

			reconstructionImage.create(sampleImage.getSize().x, sampleImage.getSize().y);

			for (int x = 0; x < sampleImage.getSize().x; x++)
				for (int y = 0; y < sampleImage.getSize().y; y++) {
					sf::Color c;

					c.r = c.g = c.b = 255.0f * (recon[x + y * sampleImage.getSize().x] - minimum) / (maximum - minimum);
					c.a = 255;

					reconstructionImage.setPixel(x, y, c);
				}

			reconstructionTexture.loadFromImage(reconstructionImage);
		}*/

		// ----------------------------- Rendering -----------------------------

		int wSize = std::pow(2 * sparseCoder.getVisibleLayerDesc(0)._radius + 1, 2);

		std::vector<float> weights(sparseCoder.getHiddenSize().x * sparseCoder.getHiddenSize().y * wSize);

		{
			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sparseCoder.getHiddenSize().x, sparseCoder.getHiddenSize().y, wSize };

			cs.getQueue().enqueueReadImage(sparseCoder.getVisibleLayer(0)._weights[neo::_back], CL_TRUE, origin, region, 0, 0, weights.data());
		}

		float minWeight = 9999.0f;
		float maxWeight = -9999.0f;

		float averageWeight = 0.0f;
		float count = 0.0f;

		for (int i = 0; i < weights.size(); i++) {
			float w = weights[i];

			minWeight = std::min(minWeight, w);
			maxWeight = std::max(maxWeight, w);

			averageWeight += w;
			count++;
		}

		int dim = 2 * sparseCoder.getVisibleLayerDesc(0)._radius + 1;

		sf::Image receptiveFieldsImage;
		receptiveFieldsImage.create(codeWidth * dim, codeHeight * dim);

		float scalar = 1.0f / (maxWeight - minWeight);

		averageWeight /= count;

		for (int sx = 0; sx < codeWidth; sx++)
			for (int sy = 0; sy < codeHeight; sy++) {
				for (int x = 0; x < dim; x++)
					for (int y = 0; y < dim; y++) {
						sf::Color color;

						color.r = color.b = color.g = 255 * (weights[sx + sy * codeWidth + (codeWidth * codeHeight) * (x + y * dim)] - minWeight) / (maxWeight - minWeight);
						color.a = 255;

						receptiveFieldsImage.setPixel(sx * dim + x, sy * dim + y, color);
					}
			}

		sf::Texture receptiveFieldsTexture;
		receptiveFieldsTexture.loadFromImage(receptiveFieldsImage);

		sf::Sprite receptiveFieldsSprite;
		receptiveFieldsSprite.setTexture(receptiveFieldsTexture);

		float scale = static_cast<float>(renderWindow.getSize().y) / static_cast<float>(receptiveFieldsImage.getSize().y);

		receptiveFieldsSprite.setScale(sf::Vector2f(scale, scale));

		renderWindow.draw(receptiveFieldsSprite);

		std::vector<float> codes(codeWidth * codeHeight);

		{
			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sparseCoder.getHiddenSize().x, sparseCoder.getHiddenSize().y, 1 };

			cs.getQueue().enqueueReadImage(sparseCoder.getHiddenStates()[neo::_back], CL_TRUE, origin, region, 0, 0, codes.data());
		}
		
		for (int sx = 0; sx < codeWidth; sx++)
			for (int sy = 0; sy < codeHeight; sy++) {
				if (codes[sx + sy * codeWidth] != 0.0f) {
					sf::RectangleShape rs;

					rs.setPosition(sx * dim * scale, sy * dim * scale);
					rs.setOutlineColor(sf::Color::Red);
					rs.setFillColor(sf::Color::Transparent);
					rs.setOutlineThickness(2.0f);

					rs.setSize(sf::Vector2f(dim * scale, dim * scale));

					renderWindow.draw(rs);
				}
			}

		sf::Sprite sampleSprite;
		sampleSprite.setTexture(sampleTexture);

		sampleSprite.setPosition(sf::Vector2f(renderWindow.getSize().x - sampleImage.getSize().x, 0.0f));

		sampleSprite.setScale(0.5f, 0.5f);

		renderWindow.draw(sampleSprite);

		sf::Sprite reconstructionSprite;
		reconstructionSprite.setTexture(reconstructionTexture);

		reconstructionSprite.setPosition(sf::Vector2f(renderWindow.getSize().x - reconstructionImage.getSize().x * 4.0f, renderWindow.getSize().y - reconstructionImage.getSize().y * 4.0f));

		reconstructionSprite.setScale(2.0f, 2.0f);

		renderWindow.draw(reconstructionSprite);

		renderWindow.display();
	} while (!quit);

	return 0;
}

#endif
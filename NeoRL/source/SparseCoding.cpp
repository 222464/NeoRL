#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_SPARSE_CODING

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/ComparisonSparseCoder.h>

#include <time.h>
#include <iostream>
#include <random>

float sig(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);
	
	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	const int sampleWidth = 16;
	const int sampleHeight = 16;
	const int codeWidth = 16;
	const int codeHeight = 16;
	const int stepsPerFrame = 5;

	// --------------------------- Create the Sparse Coder ---------------------------

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), sampleWidth, sampleHeight);
	cl::Image2D rewardImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), codeWidth, codeHeight);

	cs.getQueue().enqueueFillImage(rewardImage, cl_float4 { 1.0f, 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(codeWidth), static_cast<cl::size_type>(codeHeight), 1 });

	neo::ComparisonSparseCoder sparseCoder;

	std::vector<neo::ComparisonSparseCoder::VisibleLayerDesc> layerDescs(1);

	layerDescs[0]._size = { sampleWidth, sampleHeight };
	layerDescs[0]._radius = 8;
	layerDescs[0]._useTraces = true;

	sparseCoder.createRandom(cs, prog, layerDescs, { codeWidth, codeHeight }, 8, { -0.01f, 0.01f }, 0.0f, generator);

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

					inputf[x + y * sampleWidth] = sampleImage.getPixel(tx, ty).r / 255.0f;// +noiseDist(generator);
				}

			// Normalize
			float average = 0.0f;

			for (int i = 0; i < inputf.size(); i++) {
				average += inputf[i];
			}

			average /= inputf.size();

			float variance = 0.0f;

			for (int i = 0; i < inputf.size(); i++) {
				inputf[i] -= average;

				variance += inputf[i] * inputf[i];
			}

			variance /= inputf.size();

			variance = std::sqrt(variance);

			for (int i = 0; i < inputf.size(); i++) {
				inputf[i] /= variance;
			}

			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sampleWidth, sampleHeight, 1 };

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, origin, region, 0, 0, inputf.data());

			sparseCoder.activate(cs, std::vector<cl::Image2D>(1, inputImage), 0.02f);

			sparseCoder.learn(cs, rewardImage, std::vector<cl::Image2D>(1, inputImage), 0.1f, 0.1f);
		}

		/*if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sampleWidth, sampleHeight, 1 };

			std::vector<float> recon(sampleWidth * sampleHeight);

			cs.getQueue().enqueueReadImage(sparseCoder.getVisibleLayer(0)._reconstructionError, CL_TRUE, origin, region, 0, 0, recon.data());

			for (int x = 0; x < sampleWidth; x++)
				for (int y = 0; y < sampleHeight; y++) {
					sf::Color c = sf::Color::White;

					c.r = c.b = c.g = sig(10.0f * recon[x + y * sampleWidth]) * 255.0f;

					reconstructionImage.setPixel(x, y, c);
				}

			reconstructionTexture.loadFromImage(reconstructionImage);
		}*/

		// ----------------------------- Rendering -----------------------------

		int wSize = std::pow(2 * sparseCoder.getVisibleLayerDesc(0)._radius + 1, 2);

		std::vector<float> weights(sparseCoder.getHiddenSize().x * sparseCoder.getHiddenSize().y * wSize * 2);

		{
			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sparseCoder.getHiddenSize().x, sparseCoder.getHiddenSize().y, wSize };

			cs.getQueue().enqueueReadImage(sparseCoder.getVisibleLayer(0)._weights[neo::_back], CL_TRUE, origin, region, 0, 0, weights.data());
		}

		float minWeight = 9999.0f;
		float maxWeight = -9999.0f;

		float averageWeight = 0.0f;
		float count = 0.0f;

		for (int i = 0; i < weights.size() / 2; i++) {
			float w = weights[i * 2];

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

						color.r = color.b = color.g = 255 * (weights[2 * (sx + sy * codeWidth + (codeWidth * codeHeight) * (x + y * dim))] - minWeight) / (maxWeight - minWeight);
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
				if (codes[sx + sy * codeWidth] > 0.0f) {
					sf::RectangleShape rs;

					rs.setPosition(sx * dim * scale, sy * dim * scale);
					rs.setOutlineColor(sf::Color::Red);
					rs.setFillColor(sf::Color::Transparent);
					rs.setOutlineThickness(2.0f);

					rs.setSize(sf::Vector2f(dim * scale, dim * scale));

					renderWindow.draw(rs);
				}
			}

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
#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_SPARSE_CODING

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/ComparisonSparseCoder.h>
#include <neo/ImageWhitener.h>

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

	const int sampleWidth = 12;
	const int sampleHeight = 12;
	const int codeWidth = 20;
	const int codeHeight = 20;
	const int stepsPerFrame = 4;

	// --------------------------- Create the Sparse Coder ---------------------------

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), sampleWidth, sampleHeight);
	cl::Image2D rewardImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), codeWidth, codeHeight);
	cl::Image2D reconstruction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), sampleWidth, sampleWidth);

	cs.getQueue().enqueueFillImage(rewardImage, cl_float4 { 1.0f, 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(codeWidth), static_cast<cl::size_type>(codeHeight), 1 });

	neo::ComparisonSparseCoder sparseCoder;

	std::vector<neo::ComparisonSparseCoder::VisibleLayerDesc> layerDescs(1);

	layerDescs[0]._size = { sampleWidth, sampleHeight };
	layerDescs[0]._radius = 6;
	layerDescs[0]._useTraces = false;
	layerDescs[0]._weightAlpha = 0.01f;

	/*layerDescs[1]._size = { codeWidth, codeHeight };
	layerDescs[1]._radius = 6;
	layerDescs[1]._useTraces = false;
	layerDescs[1]._weightAlpha = 0.001f;
	layerDescs[1]._ignoreMiddle = true;*/

	sparseCoder.createRandom(cs, prog, layerDescs, { codeWidth, codeHeight }, 8, { -1.0f, 1.0f }, generator);

	// ------------------------------- Load Resources --------------------------------

	sf::Image sampleImage;

	sampleImage.loadFromFile("testIm.png");

	neo::ImageWhitener whitener;
	whitener.create(cs, prog, cl_int2{ static_cast<cl_int>(sampleImage.getSize().x), static_cast<cl_int>(sampleImage.getSize().y) }, CL_RGBA, CL_FLOAT);

	cl::Image2D sourceImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), sampleImage.getSize().x, sampleImage.getSize().y);

	std::vector<cl_float4> colors(sampleImage.getSize().x * sampleImage.getSize().y);

	for (int x = 0; x < sampleImage.getSize().x; x++)
		for (int y = 0; y < sampleImage.getSize().y; y++) {
			sf::Color c = sampleImage.getPixel(x, y);

			cl_float4 rgb;

			rgb.x = c.r / 255.0f;
			rgb.y = c.g / 255.0f;
			rgb.z = c.b / 255.0f;
			rgb.w = 1.0f;

			colors[x + y * sampleImage.getSize().x] = rgb;
		}

	cs.getQueue().enqueueWriteImage(sourceImage, CL_TRUE, { 0, 0, 0 }, { sampleImage.getSize().x, sampleImage.getSize().y, 1 }, 0, 0, colors.data());

	whitener.filter(cs, sourceImage, 3, 4000.0f);

	cs.getQueue().enqueueReadImage(whitener.getResult(), CL_TRUE, { 0, 0, 0 }, { sampleImage.getSize().x, sampleImage.getSize().y, 1 }, 0, 0, colors.data());

	sf::Image whitenedImage;
	whitenedImage.create(sampleImage.getSize().x, sampleImage.getSize().y);

	for (int x = 0; x < sampleImage.getSize().x; x++)
		for (int y = 0; y < sampleImage.getSize().y; y++) {
			cl_float4 rgb = colors[x + y * sampleImage.getSize().x];

			sf::Color c;

			c.r = (rgb.x * 0.5f + 0.5f) * 255.0f;
			c.g = (rgb.y * 0.5f + 0.5f) * 255.0f;
			c.b = (rgb.z * 0.5f + 0.5f) * 255.0f;

			whitenedImage.setPixel(x, y, c);
		}

	whitenedImage.saveToFile("whitenedTestImg.png");

	sf::Texture sampleTexture;

	sampleTexture.loadFromImage(whitenedImage);

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

					inputf[x + y * sampleWidth] = whitenedImage.getPixel(tx, ty).r / 255.0f * 2.0f - 1.0f;// +noiseDist(generator);
				}

			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sampleWidth, sampleHeight, 1 };

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, origin, region, 0, 0, inputf.data());

			std::vector<cl::Image2D> visibleStates = { inputImage, sparseCoder.getHiddenStates()[neo::_back] };

			sparseCoder.activate(cs, visibleStates, 0.1f);

			sparseCoder.learn(cs, visibleStates, 0.01f, 0.1f);
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
			cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
			cl::array<cl::size_type, 3> region = { sampleWidth, sampleHeight, 1 };

			std::vector<float> recon(sampleWidth * sampleHeight);

			sparseCoder.reconstruct(cs, sparseCoder.getHiddenStates()[neo::_back], 0, reconstruction);

			cs.getQueue().enqueueReadImage(reconstruction, CL_TRUE, origin, region, 0, 0, recon.data());

			reconstructionImage.create(sampleWidth, sampleWidth);

			for (int x = 0; x < sampleWidth; x++)
				for (int y = 0; y < sampleHeight; y++) {
					sf::Color c = sf::Color::White;

					c.r = c.b = c.g = sig(1.0f * recon[x + y * sampleWidth]) * 255.0f;

					reconstructionImage.setPixel(x, y, c);
				}

			reconstructionTexture.loadFromImage(reconstructionImage);
		}

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

		for (int i = 0; i < weights.size() / 1; i++) {
			float w = weights[i * 1];

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

		maxWeight = 1.0f;
		minWeight = -1.0f;

		for (int sx = 0; sx < codeWidth; sx++)
			for (int sy = 0; sy < codeHeight; sy++) {
				for (int x = 0; x < dim; x++)
					for (int y = 0; y < dim; y++) {
						sf::Color color;

						color.r = color.b = color.g = 255 * (weights[1 * (sx + sy * codeWidth + (codeWidth * codeHeight) * (x + y * dim))] - minWeight) / (maxWeight - minWeight);
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

		reconstructionSprite.setScale(4.0f, 4.0f);

		renderWindow.draw(reconstructionSprite);

		renderWindow.display();
	} while (!quit);

	return 0;
}

#endif
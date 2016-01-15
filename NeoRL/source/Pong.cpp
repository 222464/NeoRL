#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_PONG

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <runner/Runner.h>

#include <neo/AgentSPG.h>

#include <time.h>
#include <iostream>
#include <random>

const float ballSpeed = 0.08f;
const float ballRadius = 0.05f;
const float bottomRatio = 0.05f;
const float paddleWidthRatio = 0.1f;

sf::Vector2f _ballPosition;
sf::Vector2f _ballVelocity;

float _paddlePosition;

float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

void renderScene(sf::RenderTarget &rt) {
	sf::Vector2f size = sf::Vector2f(rt.getSize().x, rt.getSize().y);

	{
		sf::RectangleShape r;

		r.setFillColor(sf::Color::White);
		r.setSize(sf::Vector2f(ballRadius * size.x * 2.0f, ballRadius * size.y * 2.0f));

		r.setOrigin(r.getSize() * 0.5f);
		r.setPosition(_ballPosition.x * size.x, _ballPosition.y * size.y);

		rt.draw(r);
	}

	{
		sf::RectangleShape r;

		r.setFillColor(sf::Color::White);
		r.setSize(sf::Vector2f(paddleWidthRatio * size.x * 2.0f, bottomRatio * size.y));

		r.setOrigin(r.getSize() * 0.5f);
		r.setPosition(_paddlePosition * size.x, (1.0f - bottomRatio * 0.5f) * size.y);

		rt.draw(r);
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	_ballPosition = sf::Vector2f(0.5f, 0.5f);
	_ballVelocity = sf::Vector2f(0.44f, 0.55f);

	_ballVelocity *= ballSpeed / std::sqrt(_ballVelocity.x * _ballVelocity.x + _ballVelocity.y * _ballVelocity.y);

	_paddlePosition = 0.5f;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 800), "BIDInet", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	sf::RenderTexture visionRT;

	visionRT.create(16, 16);

	int inWidth = 16;
	int inHeight = 16;

	int aWidth = 2;
	int aHeight = 2;

	std::vector<neo::AgentSPG::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 8, 8 };
	layerDescs[0]._alpha = { 1.0f, 0.01f };
	layerDescs[1]._size = { 8, 8 };
	layerDescs[2]._size = { 8, 8 };

	neo::AgentSPG agent;

	agent.createRandom(cs, prog, { inWidth, inHeight }, { aWidth, aHeight }, 12, layerDescs, { -0.01f, 0.01f }, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inWidth, inHeight);
	cl::Image2D actionTaken = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), aWidth, aHeight);

	std::vector<float> input(inWidth * inHeight, 0.0f);
	std::vector<float> action(aWidth * aHeight, 0.0f);

	// ---------------------------- Game Loop -----------------------------

	std::vector<sf::Texture> layerTextures(layerDescs.size());

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float averageReward = 0.0f;
	const float averageRewardDecay = 0.003f;

	int steps = 0;

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

		visionRT.clear();

		renderScene(visionRT);

		visionRT.display();

		sf::Image img = visionRT.getTexture().copyToImage();

		for (int x = 0; x < img.getSize().x; x++)
			for (int y = 0; y < img.getSize().y; y++) {
				sf::Color c = img.getPixel(x, y);

				/*float valR = 0.0f;
				float valG = 0.0f;

				if (c.r > 0)
				valR = 1.0f;

				if (c.g > 0)
				valG = 1.0f;

				swarm.setState(x, y, 0, valR);
				swarm.setState(x, y, 1, valG);*/

				float val = 0.0f;

				if (c.r > 0)
					val = 0.5f;

				if (c.g > 0)
					val = 1.0f;

				input[x + y * inWidth] = val;
			}

		float reward = 0.0f;

		if (_ballPosition.x < 0.0f) {
			_ballPosition.x = 0.0f;

			_ballVelocity.x *= -1.0f;
		}

		if (_ballPosition.y < 0.0f) {
			_ballPosition.y = 0.0f;

			_ballVelocity.y *= -1.0f;
		}

		if (_ballPosition.x > 1.0f) {
			_ballPosition.x = 1.0f;

			_ballVelocity.x *= -1.0f;
		}

		if (_ballPosition.y > 1.0f - bottomRatio) {
			_ballPosition.y = 1.0f - bottomRatio;

			if (_ballPosition.x > _paddlePosition - paddleWidthRatio && _ballPosition.x < _paddlePosition + paddleWidthRatio) {
				reward += 1.0f;
			}
			else
				reward -= 0.5f;

			_ballVelocity.y *= -1.0f;
		}

		_ballPosition += _ballVelocity;

		averageReward = (1.0f - averageRewardDecay) * averageReward + averageRewardDecay * reward;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inWidth), static_cast<cl::size_type>(inHeight), 1 }, 0, 0, input.data());
		cs.getQueue().enqueueWriteImage(actionTaken, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, action.data());

		agent.simStep(cs, reward, inputImage, actionTaken, generator);

		std::vector<float> actionTemp(action.size() * 2);

		cs.getQueue().enqueueReadImage(agent.getAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, actionTemp.data());

		action[0] = actionTemp[0];

		if (dist01(generator) < 0.05f)
			action[0] = dist01(generator);

		action[1] = actionTemp[0] * 2.1f - 0.24f;
		action[2] = actionTemp[0] * -0.5f, +0.2f;
		action[3] = actionTemp[0] - 2.0f;

		float act = action[0] * 2.0f - 1.0f;

		_paddlePosition = std::min(1.0f, std::max(0.0f, _paddlePosition + 0.2f * std::min(1.0f, std::max(-1.0f, act))));

		std::cout << act << std::endl;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			window.clear();

			renderScene(window);

			sf::Sprite vis;

			vis.setTexture(visionRT.getTexture());

			vis.setScale(4.0f, 4.0f);

			window.draw(vis);

			/*sf::Image predImg;

			predImg.create(inWidth, inHeight);

			for (int x = 0; x < inWidth; x++)
				for (int y = 0; y < inHeight; y++) {
					sf::Color c = sf::Color::White;

					c.r = c.g = c.b = 255.0f * std::min(1.0f, std::max(0.0f, agent.getPrediction(x, y)));

					predImg.setPixel(x, y, c);
				}

			sf::Texture t;
			t.loadFromImage(predImg);

			sf::Sprite s;

			s.setTexture(t);

			s.setScale(4.0f, 4.0f);

			s.setPosition(4.0f * 16.0f, 0.0f);

			window.draw(s);*/

			float xOffset = 0.0f;
			float scale = 4.0f;

			for (int l = 0; l < layerDescs.size() - 1; l++) {
				std::vector<float> data(layerDescs[l]._size.x * layerDescs[l]._size.y);

				cs.getQueue().enqueueReadImage(agent.getLayer(l)._sc.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(layerDescs[l]._size.x), static_cast<cl::size_type>(layerDescs[l]._size.y), 1 }, 0, 0, data.data());

				sf::Image img;

				img.create(layerDescs[l]._size.x, layerDescs[l]._size.y);

				for (int x = 0; x < img.getSize().x; x++)
					for (int y = 0; y < img.getSize().y; y++) {
						sf::Color c = sf::Color::White;

						c.r = c.b = c.g = 255.0f * sigmoid(10.0f * (data[(x + y * img.getSize().x)]));

						img.setPixel(x, y, c);
					}

				layerTextures[l].loadFromImage(img);

				sf::Sprite s;

				s.setTexture(layerTextures[l]);

				s.setPosition(xOffset, window.getSize().y - img.getSize().y * scale);

				s.setScale(scale, scale);

				window.draw(s);

				xOffset += img.getSize().x * scale;
			}

			window.display();
		}

		if (steps % 100 == 0)
			std::cout << "Steps: " << steps << " Average Reward: " << averageReward << std::endl;

		//dt = clock.getElapsedTime().asSeconds();

		steps++;

	} while (!quit);

	return 0;
}

#endif
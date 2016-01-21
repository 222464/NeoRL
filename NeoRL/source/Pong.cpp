#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_PONG

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <runner/Runner.h>

#include <neo/AgentPredQ.h>
#include <neo/AgentER.h>

#include <vis/Plot.h>

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

	prog.loadFromFile("resources/neoKernels2.cl", cs);

	_ballPosition = sf::Vector2f(0.5f, 0.5f);
	_ballVelocity = sf::Vector2f(0.44f, 0.55f);

	_ballVelocity *= ballSpeed / std::sqrt(_ballVelocity.x * _ballVelocity.x + _ballVelocity.y * _ballVelocity.y);

	_paddlePosition = 0.5f;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 800), "BIDInet", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	vis::Plot plot;
	plot._curves.resize(1);
	plot._curves[0]._shadow = 0.1f;	// input

	sf::RenderTexture plotRT;
	plotRT.create(800, 800, false);

	sf::Texture lineGradient;
	lineGradient.loadFromFile("resources/lineGradient.png");

	sf::Font tickFont;
	tickFont.loadFromFile("resources/arial.ttf");

	float minCurve = -0.1f;
	float maxCurve = 0.1f;

	plotRT.setActive();
	plotRT.clear(sf::Color::White);

	const int plotSampleTicks = 1;

	const int maxBufferSize = 200;

	sf::RenderTexture visionRT;

	visionRT.create(16, 16);

	int inWidth = 16;
	int inHeight = 16;

	int aWidth = 2;
	int aHeight = 2;

	int qWidth = 4;
	int qHeight = 4;

	std::vector<neo::AgentPredQ::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 16, 16 };
	layerDescs[2]._size = { 16, 16 };

	neo::AgentPredQ agent;

	agent.createRandom(cs, prog, { inWidth, inHeight }, { aWidth, aHeight }, { qWidth, qHeight }, layerDescs, { -0.1f, 0.1f }, generator);

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

	int plotTimer = 0;

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

			if (_ballPosition.x > _paddlePosition - paddleWidthRatio - ballRadius && _ballPosition.x < _paddlePosition + paddleWidthRatio + ballRadius) {
				reward += 1.0f;
			}
			else
				reward -= 0.5f;

			_ballVelocity.y *= -1.0f;
		}

		_ballPosition += _ballVelocity;

		//reward = _paddlePosition;

		averageReward = (1.0f - averageRewardDecay) * averageReward + averageRewardDecay * reward;

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inWidth), static_cast<cl::size_type>(inHeight), 1 }, 0, 0, input.data());
		cs.getQueue().enqueueWriteImage(actionTaken, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, action.data());

		agent.simStep(cs, reward, inputImage, actionTaken);

		std::vector<float> actionTemp(action.size());

		cs.getQueue().enqueueReadImage(agent.getAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, actionTemp.data());

		action[0] = std::min(1.0f, std::max(-1.0f, actionTemp[0]));

		if (dist01(generator) < 0.05f)
			action[0] = dist01(generator) * 2.0f - 1.0f;

		action[1] = action[0] * 2.1f - 0.24f;
		action[2] = action[0] * -0.5f + 0.2f;
		action[3] = action[0] - 1.0f;

		float act = action[0];

		_paddlePosition = std::min(1.0f, std::max(0.0f, _paddlePosition + 0.2f * std::min(1.0f, std::max(-1.0f, act))));

		//std::cout << act << std::endl;

		// Plot target data
		if (plotTimer > 60) {
			plotTimer = 0;

			vis::Point p;
			p._position.x = steps / 60.0f;
			p._position.y = averageReward;
			p._color = sf::Color::Red;
			plot._curves[0]._points.push_back(p);

			if (plot._curves[0]._points.size() > maxBufferSize) {
				plot._curves[0]._points.erase(plot._curves[0]._points.begin());

				int firstIndex = 0;

				for (std::vector<vis::Point>::iterator it = plot._curves[0]._points.begin(); it != plot._curves[0]._points.end(); ++it, ++firstIndex)
					(*it)._position.x = firstIndex;
			}

			minCurve = std::min(minCurve, averageReward * 1.2f);
			maxCurve = std::max(maxCurve, averageReward * 1.2f);

			plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, plot._curves[0]._points.size()), sf::Vector2f(minCurve, maxCurve), sf::Vector2f(64.0f, 64.0f), sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxCurve - minCurve) / 10.0f), 2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);
		
			plotRT.display();
		}

		plotTimer++;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			window.clear();

			renderScene(window);

			sf::Sprite vis;

			vis.setTexture(visionRT.getTexture());

			vis.setScale(4.0f, 4.0f);

			window.draw(vis);

			sf::Sprite ps;

			ps.setTexture(plotRT.getTexture());
			ps.setPosition(800.0f, 0.0f);

			window.draw(ps);

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

			for (int l = 0; l < layerDescs.size() - 2; l++) {
				std::vector<float> data(layerDescs[l]._size.x * layerDescs[l]._size.y);

				cs.getQueue().enqueueReadImage(agent.getLayer(l + 1)._sp.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(layerDescs[l]._size.x), static_cast<cl::size_type>(layerDescs[l]._size.y), 1 }, 0, 0, data.data());

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
#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_PONG

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <runner/Runner.h>

#include <neo/AgentCACLA.h>
#include <neo/AgentQRoute.h>

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
	int inHeight = 18;

	std::vector<neo::AgentCACLA::LayerDesc> layerDescs(2);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 16, 16 };

	neo::AgentCACLA agent;

	std::vector<neo::AgentCACLA::InputType> inputTypes(inWidth * inHeight, neo::AgentCACLA::_state);

	for (int i = 0; i < inWidth; i++) {
		inputTypes[i + (inHeight - 2) * inWidth] = neo::AgentCACLA::_action;
		inputTypes[i + (inHeight - 1) * inWidth] = neo::AgentCACLA::_q;
	}

	agent.createRandom(cs, prog, { inWidth, inHeight }, 8, inputTypes, layerDescs, { -0.01f, 0.01f }, { 0.01f, 0.05f }, 0.1f, { -0.01f, 0.01f }, { -0.01f, 0.01f }, generator);

	// ---------------------------- Game Loop -----------------------------

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

				agent.setState(x, y, val);
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
				reward += 10.0f;
			}
			else
				reward -= 5.0f;

			_ballVelocity.y *= -1.0f;
		}

		_ballPosition += _ballVelocity;

		averageReward = (1.0f - averageRewardDecay) * averageReward + averageRewardDecay * reward;

		agent.simStep(reward, cs, generator);

		float act = 0.0f;

		for (int i = 4; i < 5; i++) {
			act += agent.getAction(i, 16);
		}

		_paddlePosition = std::min(1.0f, std::max(0.0f, _paddlePosition + 0.1f * std::min(1.0f, std::max(-1.0f, act))));

		//std::cout << averageReward << std::endl;

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			window.clear();

			renderScene(window);

			sf::Sprite vis;

			vis.setTexture(visionRT.getTexture());

			vis.setScale(4.0f, 4.0f);

			window.draw(vis);

			sf::Image predImg;

			predImg.create(16, 17);

			for (int x = 0; x < 16; x++)
				for (int y = 0; y < 17; y++) {
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

			window.draw(s);

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
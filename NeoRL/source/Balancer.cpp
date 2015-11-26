#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_BALANCER

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <runner/Runner.h>

#include <neo/AgentCACLA.h>
#include <neo/AgentQRoute.h>
#include <neo/AgentSwarm.h>

#include <time.h>
#include <iostream>
#include <random>

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_cpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 800), "BIDInet", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	int inWidth = 2;
	int inHeight = 2;

	int aWidth = 2;
	int aHeight = 2;

	std::vector<neo::AgentSwarm::LayerDesc> layerDescs(1);

	layerDescs[0]._size = { 8, 8 };
	//layerDescs[1]._size = { 8, 8 };

	neo::AgentSwarm agent;

	agent.createRandom(cs, prog, { inWidth, inHeight }, { aWidth, aHeight }, 6, 8, 4, layerDescs, { -0.01f, 0.01f }, { 0.01f, 0.05f }, 0.1f, { -0.01f, 0.01f }, { -0.01f, 0.01f }, generator);

	// ---------------------------- Game Loop -----------------------------

	std::vector<sf::Texture> layerTextures(layerDescs.size());

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float averageReward = 0.0f;
	const float averageRewardDecay = 0.003f;

	int steps = 0;

	float action = 0.0f;
	float ballPosition = 0.5f;
	float ballVelocity = 0.0f;

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

		ballVelocity += 0.001f * ballPosition + 0.03f * action;
		
		ballPosition += ballVelocity;

		if (ballPosition > 1.0f || ballPosition < -1.0f)
			ballVelocity = 0.0f;

		ballPosition = std::min(1.0f, std::max(-1.0f, ballPosition));

		float reward = -ballPosition * ballPosition;

		averageReward = (1.0f - averageRewardDecay) * averageReward + averageRewardDecay * reward;

		agent.setState(0, ballPosition);
		agent.setState(1, ballPosition + 1.0f);
		agent.setState(2, 1.0f - ballPosition);
		agent.setState(3, 2.0f * ballPosition);

		agent.simStep(reward, cs, generator);

		float act = 0.0f;

		for (int i = 0; i < 4; i++) {
			act += agent.getAction(i);
		}

		action += 0.1f * (std::min(1.0f, std::max(-1.0f, act)) - action);

		//std::cout << averageReward << std::endl;

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			window.clear();

			sf::ConvexShape fulcrum;

			fulcrum.setPointCount(3);

			fulcrum.setPoint(0, sf::Vector2f(-16.0f, 16.0f));
			fulcrum.setPoint(1, sf::Vector2f(16.0f, 16.0f));
			fulcrum.setPoint(2, sf::Vector2f(0.0f, 0.0f));

			fulcrum.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f + 128.0f);

			window.draw(fulcrum);

			sf::RectangleShape plank;
			plank.setSize(sf::Vector2f(256.0f, 4.0f));
			plank.setOrigin(sf::Vector2f(256.0f, 4.0f) * 0.5f);

			plank.setPosition(fulcrum.getPosition());

			plank.setRotation(30.0f * action);

			window.draw(plank);

			sf::CircleShape ball;
			ball.setRadius(8.0f);
			ball.setOrigin(8.0f, 8.0f);
			ball.setPosition(fulcrum.getPosition() + sf::Vector2f(std::cos(plank.getRotation() * 3.141596f / 180.0f) * ballPosition * plank.getSize().x * 0.5f, -ball.getRadius() + std::sin(plank.getRotation() * 3.141596f / 180.0f) * ballPosition * plank.getSize().x * 0.5f));

			window.draw(ball);

			sf::Image predImg;

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

			window.draw(s);

			float xOffset = 0.0f;
			float scale = 4.0f;

			for (int l = 0; l < layerDescs.size(); l++) {
				std::vector<float> data(layerDescs[l]._size.x * layerDescs[l]._size.y);

				cs.getQueue().enqueueReadImage(agent.getLayer(l)._scHiddenStatesPrev, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(layerDescs[l]._size.x), static_cast<cl::size_type>(layerDescs[l]._size.y), 1 }, 0, 0, data.data());

				sf::Image img;

				img.create(layerDescs[l]._size.x, layerDescs[l]._size.y);

				for (int x = 0; x < img.getSize().x; x++)
					for (int y = 0; y < img.getSize().y; y++) {
						sf::Color c = sf::Color::White;

						c.r = c.b = c.g = 255.0f * data[x + y * img.getSize().x];

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
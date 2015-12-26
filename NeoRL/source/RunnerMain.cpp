#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_RUNNER

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <runner/Runner.h>

#include <neo/AgentSwarm.h>
#include <neo/AgentSPG.h>

#include <time.h>
#include <iostream>
#include <random>

float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

int main() {
	sf::RenderWindow window;

	sf::ContextSettings glContextSettings;
	glContextSettings.antialiasingLevel = 4;

	window.create(sf::VideoMode(800, 600), "BIDInet", sf::Style::Default, glContextSettings);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// Physics
	std::shared_ptr<b2World> world = std::make_shared<b2World>(b2Vec2(0.0f, -9.81f));

	const float pixelsPerMeter = 256.0f;

	const float groundWidth = 5000.0f;
	const float groundHeight = 5.0f;

	// Create ground
	b2BodyDef groundBodyDef;
	groundBodyDef.position.Set(0.0f, 0.0f);

	b2Body* groundBody = world->CreateBody(&groundBodyDef);

	b2PolygonShape groundBox;
	groundBox.SetAsBox(groundWidth * 0.5f, groundHeight * 0.5f);

	groundBody->CreateFixture(&groundBox, 0.0f);

	sf::Texture skyTexture;

	skyTexture.loadFromFile("resources/background1.png");

	skyTexture.setSmooth(true);

	sf::Texture floorTexture;
	
	floorTexture.loadFromFile("resources/floor1.png");

	floorTexture.setRepeated(true);
	floorTexture.setSmooth(true);

	Runner runner0;

	runner0.createDefault(world, b2Vec2(0.0f, 2.762f), 0.0f, 1);

	//Runner runner1;

	//runner1.createDefault(world, b2Vec2(0.0f, 2.762f), 0.0f, 2);

	//deep::FERL ferl;

	const int clockCount = 4;

	//ferl.createRandom(3 + 3 + 2 + 2 + 1 + 2 + 2 + recCount + clockCount, 3 + 3 + 2 + 2 + recCount, 32, 0.01f, generator);

	//std::vector<float> prevAction(ferl.getNumAction(), 0.0f);

	const int inputCount = 3 + 3 + 2 + 2 + 1 + 2 + 2 + clockCount + 1;
	const int outputCount = 3 + 3 + 2 + 2;
	const int qCount = 6;

	/*std::vector<deep::CSRL::LayerDesc> layerDescs(2);

	layerDescs[0]._width = 4;
	layerDescs[0]._height = 4;

	layerDescs[1]._width = 3;
	layerDescs[1]._height = 3;

	std::vector<deep::CSRL::InputType> inputTypes(7 * 7, deep::CSRL::_state);

	for (int i = 0; i < inputCount; i++)
		inputTypes[i] = deep::CSRL::_state;

	for (int i = 0; i < outputCount; i++)
		inputTypes[i + inputCount] = deep::CSRL::_action;

	prsdr.createRandom(7, 7, 8, inputTypes, layerDescs, -0.01f, 0.01f, 0.01f, 0.05f, 0.5f, generator);
	*/

	std::vector<neo::AgentSPG::LayerDesc> layerDescs(3);

	layerDescs[0]._hiddenSize = { 16, 16 };
	//layerDescs[0]._predWeightAlpha = { 0.5f, 0.01f, 0.01f };
	layerDescs[1]._hiddenSize = { 16, 16 };
	layerDescs[2]._hiddenSize = { 16, 16 };

	neo::AgentSPG agent;

	//for (int i = inputCount + outputCount; i < inputCount + outputCount + qCount; i++)
	//	inputTypes[i] = neo::AgentCACLA::_q;

	agent.createRandom(cs, prog, { 5, 5 }, { 4, 4 }, layerDescs, { -0.01f, 0.01f }, 0.0f, generator);
	
	std::vector<int> actionIndices;

	for (int i = 0; i < outputCount; i++)
		actionIndices.push_back(inputCount + i);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 5, 5);
	std::vector<float> inputs(25, 0.0f);

	std::vector<float> actions(16, 0.0f);

	// ---------------------------- Game Loop -----------------------------

	sf::View view = window.getDefaultView();

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	int steps = 0;
	
	std::vector<float> action(3 + 3 + 2 + 2);

	std::vector<sf::Texture> layerTextures(layerDescs.size());

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

		//bidinet.simStep(cs, 0.0f, 0.98f, 0.001f, 0.95f, 0.01f, 0.01f, generator);

		const float maxRunnerBodyAngle = 0.3f;
		const float runnerBodyAngleStab = 10.0f;

		std::normal_distribution<float> noiseDist(0.0f, 0.05f);

		{
			float reward;
			
			if (!sf::Keyboard::isKeyPressed(sf::Keyboard::K))
				reward = -runner0._pBody->GetLinearVelocity().x;
			else
				reward = runner0._pBody->GetLinearVelocity().x;

			std::vector<float> state;

			runner0.getStateVector(state);
			
			for (int a = 0; a < clockCount; a++)
				state.push_back(std::sin(steps / 60.0f * 2.0f * a * 2.0f * 3.141596f) * 0.5f + 0.5f);

			for (int i = 0; i < state.size(); i++)
				inputs[i] = state[i];

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 5, 5, 1 }, 0, 0, inputs.data());

			agent.simStep(cs, reward, inputImage, generator);

			cs.getQueue().enqueueReadImage(agent.getAction(), CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, actions.data());

			cs.getQueue().finish();

			//for (int i = 0; i < actions.size(); i++)
			//	actions[i] = actions[i] * 0.5f + 0.5f;

			//std::cout << std::endl;

			runner0.motorUpdate(actions, 12.0f);

			// Keep upright
			if (std::abs(runner0._pBody->GetAngle()) > maxRunnerBodyAngle)
				runner0._pBody->SetAngularVelocity(-runnerBodyAngleStab * runner0._pBody->GetAngle());
		}

		/*{
			float reward;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))
				reward = -runner1._pBody->GetLinearVelocity().x;
			else
				reward = runner1._pBody->GetLinearVelocity().x;

			std::vector<float> state;

			runner1.getStateVector(state);

			std::vector<float> action(3 + 3 + 2 + 2 + recCount);

			for (int a = 0; a < recCount; a++)
				state.push_back(prevAction[prevAction.size() - recCount + a]);

			for (int a = 0; a < clockCount; a++)
				state.push_back(std::sin(steps / 60.0f * 2.0f * a * 2.0f * 3.141596f));

			// Bias
			state.push_back(1.0f);

			//ferl.step(state, action, reward, 0.5f, 0.99f, 0.98f, 0.05f, 16, 4, 0.05f, 0.01f, 0.05f, 600, 64, 0.01f, generator);

			for (int i = 0; i < action.size(); i++)
				action[i] = action[i] * 0.5f + 0.5f;

			prevAction = action;

			runner1.motorUpdate(action, 12.0f);

			// Keep upright
			if (std::abs(runner1._pBody->GetAngle()) > maxRunnerBodyAngle)
				runner1._pBody->SetAngularVelocity(-runnerBodyAngleStab * runner1._pBody->GetAngle());
		}*/

		int subSteps = 1;

		for (int ss = 0; ss < subSteps; ss++) {
			world->ClearForces();

			world->Step(1.0f / 60.0f / subSteps, 64, 64);
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T) || steps % 200 == 1) {
			// -------------------------------------------------------------------

			//if (!sf::Keyboard::isKeyPressed(sf::Keyboard::B))
			//	view.setCenter(runner1._pBody->GetPosition().x * pixelsPerMeter, -runner1._pBody->GetPosition().y * pixelsPerMeter);
			//else
				view.setCenter(runner0._pBody->GetPosition().x * pixelsPerMeter, -runner0._pBody->GetPosition().y * pixelsPerMeter);

			// Draw sky
			sf::Sprite skySprite;
			skySprite.setTexture(skyTexture);

			window.setView(window.getDefaultView());

			window.draw(skySprite);

			window.setView(view);

			sf::RectangleShape floorShape;
			floorShape.setSize(sf::Vector2f(groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));
			floorShape.setTexture(&floorTexture);
			floorShape.setTextureRect(sf::IntRect(0, 0, groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));

			floorShape.setOrigin(sf::Vector2f(groundWidth * pixelsPerMeter * 0.5f, groundHeight * pixelsPerMeter * 0.5f));

			window.draw(floorShape);

			//runner1.renderDefault(window, sf::Color::Blue, pixelsPerMeter);
			runner0.renderDefault(window, sf::Color::Red, pixelsPerMeter);

			window.setView(window.getDefaultView());

			float xOffset = 0.0f;
			float scale = 4.0f;

			for (int l = 0; l < layerDescs.size(); l++) {
				std::vector<float> data(agent.getLayer(l)._predAction.getHiddenSize().x * agent.getLayer(l)._predAction.getHiddenSize().y * 2);

				cs.getQueue().enqueueReadImage(agent.getLayer(l)._predAction.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(agent.getLayer(l)._predAction.getHiddenSize().x), static_cast<cl::size_type>(agent.getLayer(l)._predAction.getHiddenSize().y), 1 }, 0, 0, data.data());

				sf::Image img;

				img.create(agent.getLayer(l)._predAction.getHiddenSize().x, agent.getLayer(l)._predAction.getHiddenSize().y);

				for (int x = 0; x < img.getSize().x; x++)
					for (int y = 0; y < img.getSize().y; y++) {
						sf::Color c = sf::Color::White;

						c.r = c.b = c.g = 255.0f * sigmoid(2.0f * (data[(x + y * img.getSize().x) * 2 + 1]));

						img.setPixel(x, y, c);
					}

				layerTextures[l].loadFromImage(img);

				sf::Sprite s;

				s.setTexture(layerTextures[l]);

				s.setPosition(xOffset, window.getSize().y - img.getSize().y * scale);

				s.setScale(scale, scale);

				window.draw(s);

				xOffset += img.getSize().x * scale;

				std::cout << "Q: " << data[1] << std::endl;
			}

			window.setView(view);

			window.display();
		}
		else {
			if (steps % 100 == 0)
				std::cout << "Steps: " << steps << " Distance: " << runner0._pBody->GetPosition().x << std::endl;
		}

		//dt = clock.getElapsedTime().asSeconds();

		steps++;

	} while (!quit);

	world->DestroyBody(groundBody);

	return 0;
}

#endif
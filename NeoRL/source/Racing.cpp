#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_RACING

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>

#include <neo/AgentHA.h>
#include <deep/SDRRL.h>

#include <time.h>
#include <iostream>
#include <random>

struct Car {
	sf::Vector2f _position;
	
	float _speed;

	float _rotation;

	Car()
		: _position(0.0f, 0.0f),
		_speed(0.0f),
		_rotation(0.0f)
	{}
};

float magnitude(const sf::Vector2f &v) {
	float d = v.x * v.x + v.y * v.y;

	return std::sqrt(d);
}

float rayCast(const sf::Image &mask, const sf::Vector2f &start, const sf::Vector2f &end) {
	const float castIncrement = 3.0f;

	sf::Vector2f point = start;

	int steps = magnitude(end - start) / castIncrement;

	sf::Vector2f dir = end - start;

	dir /= std::max(0.00001f, magnitude(dir));

	float d = 0.0f;

	for (int i = 0; i < steps; i++) {
		
		sf::Color c = mask.getPixel(point.x, point.y);

		if (c == sf::Color::White)
			return d;

		point += dir * castIncrement;
		d += castIncrement;
	}

	return d;
}

void getCheckpoints(const sf::Image &checkpointsImg, std::vector<sf::Vector2f> &checkpoints) {
	for (int x = 0; x < checkpointsImg.getSize().x; x++)
		for (int y = 0; y < checkpointsImg.getSize().y; y++) {
			sf::Color c = checkpointsImg.getPixel(x, y);

			if (c.a != 0) {
				if (c.r >= checkpoints.size())
					checkpoints.resize(c.r + 1);

				checkpoints[c.r] = sf::Vector2f(x, y);
			}
		}
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::RenderWindow window;

	window.create(sf::VideoMode(1000, 956), "Racing", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	int inWidth = 8;
	int inHeight = 8;

	int aWidth = 2;
	int aHeight = 2;

	std::vector<neo::AgentHA::LayerDesc> layerDescs(2);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 12, 12 };
	//layerDescs[2]._size = { 8, 8 };

	neo::AgentHA agent;

	agent.createRandom(cs, prog, { inWidth, inHeight }, { aWidth, aHeight }, 8, layerDescs, { -0.01f, 0.01f }, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inWidth, inHeight);
	std::vector<float> input(inWidth * inHeight, 0.0f);
	std::vector<float> action(aWidth * aHeight, 0.0f);

	deep::SDRRL agent2;

	agent2.createRandom(64, 4, 32, -0.01f, 0.01f, 0.01f, 0.05f, 0.1f, generator);

	// Dummy agent
	int cells = 128;

	std::vector<float> weightsA(cells);
	std::vector<float> tracesA(cells, 0.0f);
	std::vector<float> weightsQ(cells);
	std::vector<float> tracesQ(cells, 0.0f);

	std::uniform_real_distribution<float> wDist(-0.01f, 0.01f);

	for (int i = 0; i < cells; i++) {
		weightsQ[i] = wDist(generator);
		weightsA[i] = wDist(generator);
	}

	float prevVal = 0.0f;

	// -------------------------- Game Resources --------------------------

	sf::Texture backgroundTex;
	backgroundTex.loadFromFile("resources/racing/racingBackground.png");

	sf::Texture foregroundTex;
	foregroundTex.loadFromFile("resources/racing/racingForeground.png");

	sf::Image collisionImg;
	collisionImg.loadFromFile("resources/racing/racingCollision.png");

	sf::Image checkpointsImg;
	checkpointsImg.loadFromFile("resources/racing/racingCheckpoints.png");

	std::vector<sf::Vector2f> checkpoints;

	getCheckpoints(checkpointsImg, checkpoints);
	
	sf::Texture carTex;
	carTex.loadFromFile("resources/racing/racingCar.png");

	Car car;

	// Reset
	car._position = checkpoints[0];
	car._rotation = std::atan2(checkpoints[1].y - checkpoints[0].y, checkpoints[1].x - checkpoints[0].x);

	int curCheckpoint = 0;

	int laps = 0;

	float prevDistance = 0.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	std::vector<sf::Texture> layerTextures(layerDescs.size());

	sf::View view;

	view.setCenter(250, 239);

	view.zoom(0.5f);

	window.setView(view);

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

		const float maxSpeed = 5.0f;
		const float accel = 0.03f;
		const float spinRate = 0.3f;

		/*action[0] = 0.0f;
		action[1] = 0.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
			action[0] = 1.0f;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			action[0] = -1.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			action[1] = 1.0f;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			action[1] = -1.0f;*/

		// Physics update
		sf::Vector2f prevPosition = car._position;

		car._position += sf::Vector2f(std::cos(car._rotation), std::sin(car._rotation)) * car._speed;
		car._speed *= 0.95f;

		car._speed = std::min(maxSpeed, std::max(-maxSpeed, car._speed + accel));// * (action[0] * 0.5f + 0.5f)
		car._rotation = std::fmod(car._rotation + std::min(1.0f, std::max(-1.0f, action[0])) * spinRate, 3.141596f * 2.0f);

		sf::Color curColor = collisionImg.getPixel(car._position.x, car._position.y);

		bool reset = false;

		if (curColor == sf::Color::White || sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
			// Reset
			car._position = checkpoints[0];
			car._speed = 0.0f;
			car._rotation = std::atan2(checkpoints[1].y - checkpoints[0].y, checkpoints[1].x - checkpoints[0].x);
			curCheckpoint = 0;
			laps = 0;
			prevDistance = 0.0f;

			reset = true;
		}

		sf::Vector2f vec = checkpoints[(curCheckpoint + 1) % static_cast<int>(checkpoints.size())] - checkpoints[curCheckpoint];

		// Project position onto vec
		sf::Vector2f relPos = car._position - checkpoints[curCheckpoint];

		sf::Vector2f proj = (relPos.x * vec.x + relPos.y * vec.y) / std::pow(magnitude(vec), 2) * vec;

		float addDist = magnitude(proj) * ((vec.x * proj.x + vec.y * proj.y) > 0.0f ? 1.0f : -1.0f);

		// If past checkpoint (before or after current segment)
		if (addDist >= magnitude(vec)) {
			curCheckpoint = (curCheckpoint + 1) % static_cast<int>(checkpoints.size());

			if (curCheckpoint == 0)
				laps++;
		}
		else if (addDist < 0.0f) {
			curCheckpoint = (curCheckpoint - 1) % static_cast<int>(checkpoints.size());

			if (curCheckpoint < 0) {
				curCheckpoint += checkpoints.size();
				laps--;
			}
		}

		// Re-do projection in case checkpoint changed
		vec = checkpoints[(curCheckpoint + 1) % static_cast<int>(checkpoints.size())] - checkpoints[curCheckpoint];

		// Project position onto vec
		relPos = car._position - checkpoints[curCheckpoint];

		proj = (relPos.x * vec.x + relPos.y * vec.y) / std::pow(magnitude(vec), 2) * vec;

		addDist = magnitude(proj) * ((vec.x * proj.x + vec.y * proj.y) > 0.0f ? 1.0f : -1.0f);

		// Car distance
		float distance = 0.0f;

		// Count up to current checkpoint
		for (int i = 0; i < curCheckpoint; i++)
			distance += magnitude(checkpoints[(i + 1) % static_cast<int>(checkpoints.size())] - checkpoints[i]);

		// Add laps
		float totalDist = 0.0f;

		for (int i = 0; i < checkpoints.size(); i++)
			totalDist += magnitude(checkpoints[(i + 1) % static_cast<int>(checkpoints.size())] - checkpoints[i]);

		distance += laps * totalDist;

		distance += addDist;

		float deltaDistance = reset ? 0.0f : distance - prevDistance;

		prevDistance = distance;

		sf::Vector2f carDir(car._position - prevPosition);

		carDir /= std::max(0.00001f, magnitude(carDir));

		sf::Vector2f trackDir = vec / magnitude(vec);
		sf::Vector2f trackPerp(-trackDir.y, trackDir.x);

		float reward = std::abs(car._speed) * (carDir.x * trackDir.x + carDir.y * trackDir.y);

		// Sensors
		std::vector<float> sensors(16);

		const float sensorAngle = 0.2f;
		const float sensorRange = 60.0f;

		for (int s = 0; s < sensors.size(); s++) {
			float d = sensorAngle * (s - sensors.size() * 0.5f) + car._rotation;

			sf::Vector2f dir = sf::Vector2f(std::cos(d), std::sin(d));

			sf::Vector2f begin = car._position;
			sf::Vector2f end = car._position + dir * sensorRange;

			float v = rayCast(collisionImg, begin, end);

			sensors[s] = v / sensorRange;
		}

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			sf::Sprite backgroundS;
			backgroundS.setTexture(backgroundTex);

			window.draw(backgroundS);

			sf::VertexArray va;

			va.setPrimitiveType(sf::Lines);
			va.resize(sensors.size() * 2);

			for (int s = 0; s < sensors.size(); s++) {
				float d = sensorAngle * (s - sensors.size() * 0.5f) + car._rotation;

				sf::Vector2f dir = sf::Vector2f(std::cos(d), std::sin(d));

				va[s * 2 + 0] = car._position;
				va[s * 2 + 1] = car._position + dir * sensors[s] * sensorRange;
			}

			window.draw(va);

			sf::Sprite carS;
			carS.setTexture(carTex);

			carS.setOrigin(carTex.getSize().x * 0.5f, carTex.getSize().y * 0.5f);
			carS.setPosition(car._position);
			carS.setRotation(car._rotation * 180.0f / 3.141596f + 90.0f);
			carS.setScale(0.5f, 0.5f);

			window.draw(carS);

			// Draw checkpoints if desired
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)) {
				sf::VertexArray ar;

				ar.setPrimitiveType(sf::LinesStrip);
				ar.resize(checkpoints.size() + 1);

				for (int i = 0; i < checkpoints.size(); i++) {
					ar[i] = sf::Vertex(checkpoints[i], sf::Color::Green);
				}

				ar[checkpoints.size()] = sf::Vertex(checkpoints.front(), sf::Color::Green);

				window.draw(ar);
			}

			sf::Sprite foregroundS;
			foregroundS.setTexture(foregroundTex);
			//foregroundS.setColor(sf::Color::Red);
			window.draw(foregroundS);

			float xOffset = 0.0f;
			float scale = 4.0f;

			for (int l = 0; l < layerDescs.size(); l++) {
				std::vector<float> data(layerDescs[l]._size.x * layerDescs[l]._size.y);

				cs.getQueue().enqueueReadImage(agent.getLayer(l)._sc.getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(layerDescs[l]._size.x), static_cast<cl::size_type>(layerDescs[l]._size.y), 1 }, 0, 0, data.data());

				sf::Image img;

				img.create(layerDescs[l]._size.x, layerDescs[l]._size.y);

				for (int x = 0; x < img.getSize().x; x++)
					for (int y = 0; y < img.getSize().y; y++) {
						sf::Color c = sf::Color::White;

						c.r = c.b = c.g = 255.0f * data[(x + y * img.getSize().x)];

						img.setPixel(x, y, c);
					}

				layerTextures[l].loadFromImage(img);

				sf::Sprite s;

				s.setTexture(layerTextures[l]);

				s.setPosition(xOffset, window.getSize().y * 0.5f - img.getSize().y * scale);

				s.setScale(scale, scale);

				window.draw(s);

				xOffset += img.getSize().x * scale;
			}


			window.display();
		}

		for (int i = 0; i < sensors.size(); i++)
			input[i] = sensors[i];

		for (int i = 0; i < sensors.size(); i++)
			input[i + sensors.size()] = 1.0f - sensors[i];

		input[sensors.size() * 2 + 0] = (car._rotation / (2.0f * 3.141596f));
		input[sensors.size() * 2 + 1] = 1.0f - (car._rotation / (2.0f * 3.141596f));
		input[sensors.size() * 2 + 2] = car._speed * 0.1f;

		//for (int i = 0; i < input.size(); i++)
		//	agent2.setState(i, input[i]);

		//agent2.simStep(reset ? -1.0f : 0.04f * (reward - std::abs(action[1]) * 2.0f), 0.1f, 0.98f, 0.001f, 0.01f, 0.01f, 0.01f, 0.01f, 20, 0.05f, 0.97f, 0.01f, 0.01f, 0.01f, 4.0f, generator);

		//for (int i = 0; i < action.size(); i++)
		//	action[i] = agent2.getAction(i);

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inWidth), static_cast<cl::size_type>(inHeight), 1 }, 0, 0, input.data());

		agent.simStep(cs, reset ? -1.0f : 0.02f * reward, inputImage, generator);

		cs.getQueue().enqueueReadImage(agent.getExploratoryAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, action.data());

		// Dummy agent
		/*float angle = std::acos(std::min(1.0f, std::max(-1.0f, trackPerp.x * carDir.x + trackPerp.y * carDir.y)));

		if (angle < 0.0f)
			angle += 3.141596f;

		float ratio = angle / (3.14159f);

		int index = static_cast<int>(ratio * cells);

		float q = weightsQ[index];

		float a = weightsA[index];

		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		float ae = dist01(generator) < 0.01f ? dist01(generator) : std::min(1.0f, std::max(0.0f, a));

		float r = reset ? -1.0f : 0.02f * reward;

		float tdError = r + 0.97f * q - prevVal;
		std::cout << a << std::endl;
		prevVal = q;

		for (int i = 0; i < cells; i++)
			weightsQ[i] += 0.01f * tdError * tracesQ[i];

		for (int i = 0; i < cells; i++)
			tracesQ[i] = 0.95f * tracesQ[i] + (i == index ? 1.0f : 0.0f);

		for (int i = 0; i < cells; i++)
			weightsA[i] += 0.1f * (tdError > 0.0f ? 1.0f : 0.0f) * tracesA[i];

		for (int i = 0; i < cells; i++)
			tracesA[i] = 0.95f * tracesA[i] + (i == index ? ae - a : 0.0f);

		action[0] = ae * 2.0f - 1.0f;

		action[0] = 20.0f * (ratio * 2.0f - 1.0f);*/
	} while (!quit);

	return 0;
}

#endif
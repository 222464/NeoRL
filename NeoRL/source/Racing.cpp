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
	const float castIncrement = 8.0f;

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

	cs.create(sys::ComputeSystem::_cpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::RenderWindow window;

	window.create(sf::VideoMode(640, 640), "Racing", sf::Style::Default);

	window.setFramerateLimit(60);
	window.setVerticalSyncEnabled(true);

	int inWidth = 5;
	int inHeight = 5;

	int aWidth = 2;
	int aHeight = 2;

	std::vector<neo::AgentHA::LayerDesc> layerDescs(2);

	layerDescs[0]._size = { 32, 32 };
	layerDescs[1]._size = { 16, 16 };
	//layerDescs[2]._size = { 16, 16 };

	neo::AgentHA agent;

	agent.createRandom(cs, prog, { inWidth, inHeight }, { aWidth, aHeight }, 8, layerDescs, { -0.01f, 0.01f }, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inWidth, inHeight);
	std::vector<float> input(inWidth * inHeight, 0.0f);
	std::vector<float> action(aWidth * aHeight, 0.0f);

	deep::SDRRL agent2;

	agent2.createRandom(18, 2, 32, -0.01f, 0.01f, 0.01f, 0.05f, 0.05f, generator);

	// -------------------------- Game Resources --------------------------

	sf::Texture backgroundTex;
	backgroundTex.loadFromFile("resources/racingBackground.png");

	sf::Image collisionImg;
	collisionImg.loadFromFile("resources/racingCollision.png");

	sf::Image checkpointsImg;
	checkpointsImg.loadFromFile("resources/racingCheckpoints.png");

	std::vector<sf::Vector2f> checkpoints;

	getCheckpoints(checkpointsImg, checkpoints);
	
	sf::Texture carTex;
	carTex.loadFromFile("resources/racingCar.png");

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

		const float maxSpeed = 10.0f;
		const float accel = 0.1f;
		const float spinRate = 0.1f;

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

		car._speed = std::min(maxSpeed, std::max(-maxSpeed, car._speed + accel));// *(action[0] * 0.5f + 0.5f)));
		car._rotation = std::fmod(car._rotation + std::min(1.0f, std::max(-1.0f, 0.333f * (action[0] + action[1] - action[2] - action[3]))) * spinRate, 3.141596f * 2.0f);

		sf::Color curColor = collisionImg.getPixel(car._position.x, car._position.y);

		bool reset = false;

		if (curColor == sf::Color::White) {
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
		const float sensorRange = 200.0f;

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

			window.draw(carS);

			window.display();
		}

		for (int i = 0; i < sensors.size(); i++)
			input[i] = sensors[i];

		input[sensors.size() + 0] = (car._rotation / (2.0f * 3.141596f)) * 2.0f - 1.0f;
		input[sensors.size() + 1] = car._speed * 0.1f;

		/*for (int i = 0; i < 18; i++)
			agent2.setState(i, input[i]);

		agent2.simStep(deltaDistance * 10.0f, 0.1f, 0.99f, 8, 8, 0.1f, 0.01f, 0.1f, 0.01f, 0.01f, 0.1f, 17, 0.1f, 0.95f, 0.01f, 0.01f, 0.01f, 4.0f, generator);

		action[0] = agent2.getAction(0) * 2.0f - 1.0f;
		action[1] = agent2.getAction(1) * 2.0f - 1.0f;*/

		cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inWidth), static_cast<cl::size_type>(inHeight), 1 }, 0, 0, input.data());

		agent.simStep(cs, reset ? -1.0f : 0.04f * (reward - std::abs(action[1]) * 2.0f), inputImage, generator);

		cs.getQueue().enqueueReadImage(agent.getExploratoryAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, action.data());
	} while (!quit);

	return 0;
}

#endif
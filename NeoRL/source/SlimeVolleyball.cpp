#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_SLIME_VOLLEYBALL

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/AgentSPG.h>

#include <iostream>
#include <random>

struct PhyObj {
	sf::Vector2f _position;
	sf::Vector2f _velocity;

	PhyObj()
		: _position(0.0f, 0.0f), _velocity(0.0f, 0.0f)
	{}
};

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	std::vector<neo::AgentSPG::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 64, 64 };
	layerDescs[0]._predictiveRadius = 12;
	layerDescs[0]._feedBackRadius = 12;
	layerDescs[1]._size = { 48, 48 };
	layerDescs[2]._size = { 32, 32 };

	neo::AgentSPG agent;

	int inWidth = 32;
	int inHeight = 32;
	int aWidth = 2;
	int aHeight = 2;
	int frameSkip = 3;

	agent.createRandom(cs, prog, { inWidth, inHeight }, { aWidth, aHeight }, layerDescs, { -0.05f, 0.05f }, generator);

	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inWidth, inHeight);

	sf::VideoMode videoMode(1280, 720);

	sf::RenderWindow renderWindow;

	renderWindow.create(videoMode, "Reinforcement Learning", sf::Style::Default);

	renderWindow.setVerticalSyncEnabled(true);

	renderWindow.setFramerateLimit(60);

	// --------------------------------- Game Init -----------------------------------

	const float slimeRadius = 94.5f;
	const float ballRadius = 23.5f;
	const float wallRadius = 22.5f;
	const float fieldRadius = 640.0f;

	const float gravity = 900.0f;
	const float slimeBounce = 100.0f;
	const float wallBounceDecay = 0.8f;
	const float slimeJump = 500.0f;
	const float maxSlimeSpeed = 1000.0f;
	const float slimeMoveAccel = 5000.0f;
	const float slimeMoveDeccel = 8.0f;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::Vector2f fieldCenter = sf::Vector2f(renderWindow.getSize().x * 0.5f, renderWindow.getSize().y * 0.5f + 254.0f);
	sf::Vector2f wallCenter = fieldCenter + sf::Vector2f(0.0f, -182.0f);

	PhyObj blue;
	PhyObj red;
	PhyObj ball;

	blue._position = fieldCenter + sf::Vector2f(-200.0f, 0.0f);
	blue._velocity = sf::Vector2f(0.0f, 0.0f);
	red._position = fieldCenter + sf::Vector2f(200.0f, 0.0f);
	red._velocity = sf::Vector2f(0.0f, 0.0f);
	ball._position = fieldCenter + sf::Vector2f(2.0f, -300.0f);
	ball._velocity = sf::Vector2f((dist01(generator)) * 600.0f, -(dist01(generator)) * 500.0f);

	sf::Texture backgroundTexture;
	backgroundTexture.loadFromFile("resources/slimevolleyball/background.png");

	sf::Texture blueSlimeTexture;
	blueSlimeTexture.loadFromFile("resources/slimevolleyball/slimeBodyBlue.png");

	sf::Texture redSlimeTexture;
	redSlimeTexture.loadFromFile("resources/slimevolleyball/slimeBodyRed.png");

	sf::Texture ballTexture;
	ballTexture.loadFromFile("resources/slimevolleyball/ball.png");

	sf::Texture eyeTexture;
	eyeTexture.loadFromFile("resources/slimevolleyball/slimeEye.png");
	eyeTexture.setSmooth(true);

	sf::Texture arrowTexture;
	arrowTexture.loadFromFile("resources/slimevolleyball/arrow.png");

	sf::Font scoreFont;
	scoreFont.loadFromFile("resources/slimevolleyball/scoreFont.ttf");

	int scoreRed = 0;
	int scoreBlue = 0;

	int prevScoreRed = 0;
	int prevScoreBlue = 0;

	float prevBallX = fieldCenter.x;

	sf::RenderTexture blueRT;

	blueRT.create(inWidth, inHeight);

	// ------------------------------- Simulation Loop -------------------------------

	bool quit = false;

	float dt = 0.017f;

	int skipFrameCounter = 0;

	float blueReward = 0.0f;
	float redReward = 0.0f;

	std::vector<float> actionTemp(aWidth * aHeight * 2, 0.0f);

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

		// ---------------------------------- Physics ----------------------------------

		bool blueBounced = false;
		bool redBounced = false;

		// Ball
		{
			ball._velocity.y += gravity * dt;
			ball._position += ball._velocity * dt;

			// To floor (game restart)
			if (ball._position.y + ballRadius > fieldCenter.y) {
				if (ball._position.x < fieldCenter.x)
					scoreRed++;
				else
					scoreBlue++;

				blue._position = fieldCenter + sf::Vector2f(-200.0f, 0.0f);
				blue._velocity = sf::Vector2f(0.0f, 0.0f);
				red._position = fieldCenter + sf::Vector2f(200.0f, 0.0f);
				red._velocity = sf::Vector2f(0.0f, 0.0f);
				ball._position = fieldCenter + sf::Vector2f(2.0f, -300.0f);
				ball._velocity = sf::Vector2f(((dist01(generator))) * -600.0f, -(dist01(generator)) * 500.0f);
			}

			// To wall
			if (((ball._position.x + ballRadius) > (wallCenter.x - wallRadius) && ball._position.x < wallCenter.x) || ((ball._position.x - ballRadius) < (wallCenter.x + wallRadius) && ball._position.x > wallCenter.x)) {
				// If above rounded part
				if (ball._position.y < wallCenter.y) {
					sf::Vector2f delta = ball._position - wallCenter;

					float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

					if (dist < wallRadius + ballRadius) {
						sf::Vector2f normal = delta / dist;

						// Reflect velocity
						sf::Vector2f reflectedVelocity = ball._velocity - 2.0f * (ball._velocity.x * normal.x + ball._velocity.y * normal.y) * normal;

						ball._velocity = reflectedVelocity * wallBounceDecay;

						ball._position = wallCenter + normal * (wallRadius + ballRadius);
					}
				}
				else {
					// If on left side
					if (ball._position.x < wallCenter.x) {
						ball._velocity.x = wallBounceDecay * -ball._velocity.x;
						ball._position.x = wallCenter.x - wallRadius - ballRadius;
					}
					else {
						ball._velocity.x = wallBounceDecay * -ball._velocity.x;
						ball._position.x = wallCenter.x + wallRadius + ballRadius;
					}
				}
			}

			// To blue slime			
			{
				sf::Vector2f delta = ball._position - blue._position;

				float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

				if (dist < slimeRadius + ballRadius) {
					sf::Vector2f normal = delta / dist;

					// Reflect velocity
					sf::Vector2f reflectedVelocity = ball._velocity - 2.0f * (ball._velocity.x * normal.x + ball._velocity.y * normal.y) * normal;

					float magnitude = std::sqrt(reflectedVelocity.x * reflectedVelocity.x + reflectedVelocity.y * reflectedVelocity.y);

					sf::Vector2f normalizedReflected = reflectedVelocity / magnitude;

					ball._velocity = blue._velocity + (magnitude > slimeBounce ? reflectedVelocity : normalizedReflected * slimeBounce);

					ball._position = blue._position + normal * (wallRadius + slimeRadius);

					blueBounced = true;
				}
			}

			// To red slime			
			{
				sf::Vector2f delta = ball._position - red._position;

				float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

				if (dist < slimeRadius + ballRadius) {
					sf::Vector2f normal = delta / dist;

					// Reflect velocity
					sf::Vector2f reflectedVelocity = ball._velocity - 2.0f * (ball._velocity.x * normal.x + ball._velocity.y * normal.y) * normal;

					float magnitude = std::sqrt(reflectedVelocity.x * reflectedVelocity.x + reflectedVelocity.y * reflectedVelocity.y);

					sf::Vector2f normalizedReflected = reflectedVelocity / magnitude;

					ball._velocity = red._velocity + (magnitude > slimeBounce ? reflectedVelocity : normalizedReflected * slimeBounce);

					ball._position = red._position + normal * (wallRadius + slimeRadius);

					redBounced = true;
				}
			}

			// Out of field, left and right
			{
				if (ball._position.x - ballRadius < fieldCenter.x - fieldRadius) {
					ball._velocity.x = wallBounceDecay * -ball._velocity.x;
					ball._position.x = fieldCenter.x - fieldRadius + ballRadius;
				}
				else if (ball._position.x + ballRadius > fieldCenter.x + fieldRadius) {
					ball._velocity.x = wallBounceDecay * -ball._velocity.x;
					ball._position.x = fieldCenter.x + fieldRadius - ballRadius;
				}
			}
		}

		// Blue slime
		{
			blue._velocity.y += gravity * dt;
			blue._velocity.x += -slimeMoveDeccel * blue._velocity.x * dt;
			blue._position += blue._velocity * dt;

			bool moveLeft;
			bool moveRight;
			bool jump;

			sf::Image img = blueRT.getTexture().copyToImage();

			std::vector<float> greyData(img.getSize().x * img.getSize().y);

			for (int x = 0; x < img.getSize().x; x++)
				for (int y = 0; y < img.getSize().y; y++) {
					sf::Color c = img.getPixel(x, y);
					greyData[x + y * img.getSize().x] = 0.333f * (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f);
				}
	
			if (skipFrameCounter == 0) {
				cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inWidth), static_cast<cl::size_type>(inHeight), 1 }, 0, 0, greyData.data());

				agent.simStep(cs, blueReward * 0.01f, inputImage, generator);
				
				cs.getQueue().enqueueReadImage(agent.getExploratoryAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(aWidth), static_cast<cl::size_type>(aHeight), 1 }, 0, 0, actionTemp.data());
			
				cs.getQueue().finish();
			}

			moveLeft = actionTemp[0 * 2 + 0] > 0.5f;
			moveRight = actionTemp[1 * 2 + 0] > 0.5f;
			jump = actionTemp[2 * 2 + 0] > 0.5f;

			std::cout << blueReward << " " << actionTemp[0 * 2 + 0] << " " << actionTemp[1 * 2 + 0] << std::endl;

			if (moveLeft) {
				blue._velocity.x += -slimeMoveAccel * dt;

				if (blue._velocity.x < -maxSlimeSpeed)
					blue._velocity.x = -maxSlimeSpeed;
			}
			else if (moveRight) {
				blue._velocity.x += slimeMoveAccel * dt;

				if (blue._velocity.x > maxSlimeSpeed)
					blue._velocity.x = maxSlimeSpeed;
			}

			if (blue._position.y > fieldCenter.y) {
				blue._velocity.y = 0.0f;
				blue._position.y = fieldCenter.y;

				if (jump)
					blue._velocity.y -= slimeJump;
			}

			if (blue._position.x - slimeRadius < fieldCenter.x - fieldRadius) {
				blue._velocity.x = 0.0f;
				blue._position.x = fieldCenter.x - fieldRadius + slimeRadius;
			}

			if (blue._position.x + slimeRadius > wallCenter.x - wallRadius) {
				blue._velocity.x = 0.0f;
				blue._position.x = wallCenter.x - wallRadius - slimeRadius;
			}
		}

		// Red slime
		{
			red._velocity.y += gravity * dt;
			red._velocity.x += -slimeMoveDeccel * red._velocity.x * dt;
			red._position += red._velocity * dt;

			bool moveLeft;
			bool moveRight;
			bool jump;

			moveLeft = sf::Keyboard::isKeyPressed(sf::Keyboard::Left);
			moveRight = sf::Keyboard::isKeyPressed(sf::Keyboard::Right);
			jump = sf::Keyboard::isKeyPressed(sf::Keyboard::Up);

			if (moveLeft) {
				red._velocity.x += -slimeMoveAccel * dt;

				if (red._velocity.x < -maxSlimeSpeed)
					red._velocity.x = -maxSlimeSpeed;
			}
			else if (moveRight) {
				red._velocity.x += slimeMoveAccel * dt;

				if (red._velocity.x > maxSlimeSpeed)
					red._velocity.x = maxSlimeSpeed;
			}

			if (red._position.y > fieldCenter.y) {
				red._velocity.y = 0.0f;
				red._position.y = fieldCenter.y;

				if (jump)
					red._velocity.y -= slimeJump;
			}

			if (red._position.x + slimeRadius > fieldCenter.x + fieldRadius) {
				red._velocity.x = 0.0f;
				red._position.x = fieldCenter.x + fieldRadius - slimeRadius;
			}

			if (red._position.x - slimeRadius < wallCenter.x + wallRadius) {
				red._velocity.x = 0.0f;
				red._position.x = wallCenter.x + wallRadius + slimeRadius;
			}
		}

		blueReward = scoreBlue - prevScoreBlue - (scoreRed - prevScoreRed);// -0.00005f * std::abs(ball._position.x - blue._position.x) + (blueBounced ? 0.2f : 0.0f);
		redReward = scoreRed - prevScoreRed - (scoreBlue - prevScoreBlue);// -0.00005f * std::abs(ball._position.x - red._position.x) + (redBounced ? 0.2f : 0.0f);

		prevScoreRed = scoreRed;
		prevScoreBlue = scoreBlue;
		prevBallX = ball._position.x;

		// --------------------------------- Rendering ---------------------------------

		if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			{
				renderWindow.clear();

				{
					sf::Sprite s;
					s.setTexture(backgroundTexture);
					s.setOrigin(backgroundTexture.getSize().x * 0.5f, backgroundTexture.getSize().y * 0.5f);
					s.setPosition(renderWindow.getSize().x * 0.5f, renderWindow.getSize().y * 0.5f);

					renderWindow.draw(s);
				}

				{
					sf::Sprite s;
					s.setTexture(blueSlimeTexture);
					s.setOrigin(blueSlimeTexture.getSize().x * 0.5f, blueSlimeTexture.getSize().y);
					s.setPosition(blue._position);

					renderWindow.draw(s);
				}

				{
					sf::Sprite s;
					s.setTexture(eyeTexture);
					s.setOrigin(eyeTexture.getSize().x * 0.5f, eyeTexture.getSize().y * 0.5f);
					s.setPosition(blue._position + sf::Vector2f(50.0f, -28.0f));

					sf::Vector2f delta = ball._position - s.getPosition();

					float angle = std::atan2(delta.y, delta.x);

					s.setRotation(angle * 180.0f / 3.141596f);

					renderWindow.draw(s);
				}

				{
					sf::Sprite s;
					s.setTexture(redSlimeTexture);
					s.setOrigin(redSlimeTexture.getSize().x * 0.5f, redSlimeTexture.getSize().y);
					s.setPosition(red._position);

					renderWindow.draw(s);
				}

				{
					sf::Sprite s;
					s.setTexture(eyeTexture);
					s.setOrigin(eyeTexture.getSize().x * 0.5f, eyeTexture.getSize().y * 0.5f);
					s.setPosition(red._position + sf::Vector2f(-50.0f, -28.0f));

					sf::Vector2f delta = ball._position - s.getPosition();

					float angle = std::atan2(delta.y, delta.x);

					s.setRotation(angle * 180.0f / 3.141596f);

					renderWindow.draw(s);
				}

				{
					sf::Sprite s;
					s.setTexture(ballTexture);
					s.setOrigin(ballTexture.getSize().x * 0.5f, ballTexture.getSize().y * 0.5f);
					s.setPosition(ball._position);

					renderWindow.draw(s);
				}

				if (ball._position.y + ballRadius < 0.0f) {
					sf::Sprite s;
					s.setTexture(arrowTexture);
					s.setOrigin(arrowTexture.getSize().x * 0.5f, 0.0f);
					s.setPosition(ball._position.x, 0.0f);

					renderWindow.draw(s);
				}

				{
					sf::Text scoreText;
					scoreText.setFont(scoreFont);
					scoreText.setString(std::to_string(scoreBlue));
					scoreText.setCharacterSize(100);

					float width = scoreText.getLocalBounds().width;

					scoreText.setPosition(fieldCenter.x - width * 0.5f - 100.0f, 10.0f);

					scoreText.setColor(sf::Color(100, 133, 255));

					renderWindow.draw(scoreText);
				}

				{
					sf::Text scoreText;
					scoreText.setFont(scoreFont);
					scoreText.setString(std::to_string(scoreRed));
					scoreText.setCharacterSize(100);

					float width = scoreText.getLocalBounds().width;

					scoreText.setPosition(fieldCenter.x - width * 0.5f + 100.0f, 10.0f);

					scoreText.setColor(sf::Color(255, 100, 100));

					renderWindow.draw(scoreText);
				}

				{
					sf::Sprite visionSprite;
					visionSprite.setTexture(blueRT.getTexture());
					visionSprite.scale(4.0f, 4.0f);

					renderWindow.draw(visionSprite);
				}
			}

			renderWindow.display();
		}

		blueRT.setView(renderWindow.getView());

		{
			blueRT.clear();

			{
				sf::Sprite s;
				s.setTexture(backgroundTexture);
				s.setOrigin(backgroundTexture.getSize().x * 0.5f, backgroundTexture.getSize().y * 0.5f);
				s.setPosition(renderWindow.getSize().x * 0.5f, renderWindow.getSize().y * 0.5f);

				blueRT.draw(s);
			}

			{
				sf::Sprite s;
				s.setTexture(blueSlimeTexture);
				s.setOrigin(blueSlimeTexture.getSize().x * 0.5f, blueSlimeTexture.getSize().y);
				s.setPosition(blue._position);

				blueRT.draw(s);
			}

			{
				sf::Sprite s;
				s.setTexture(eyeTexture);
				s.setOrigin(eyeTexture.getSize().x * 0.5f, eyeTexture.getSize().y * 0.5f);
				s.setPosition(blue._position + sf::Vector2f(50.0f, -28.0f));

				sf::Vector2f delta = ball._position - s.getPosition();

				float angle = std::atan2(delta.y, delta.x);

				s.setRotation(angle * 180.0f / 3.141596f);

				blueRT.draw(s);
			}

			{
				sf::Sprite s;
				s.setTexture(redSlimeTexture);
				s.setOrigin(redSlimeTexture.getSize().x * 0.5f, redSlimeTexture.getSize().y);
				s.setPosition(red._position);

				blueRT.draw(s);
			}

			{
				sf::Sprite s;
				s.setTexture(eyeTexture);
				s.setOrigin(eyeTexture.getSize().x * 0.5f, eyeTexture.getSize().y * 0.5f);
				s.setPosition(red._position + sf::Vector2f(-50.0f, -28.0f));

				sf::Vector2f delta = ball._position - s.getPosition();

				float angle = std::atan2(delta.y, delta.x);

				s.setRotation(angle * 180.0f / 3.141596f);

				blueRT.draw(s);
			}

			{
				sf::Sprite s;
				s.setTexture(ballTexture);
				s.setOrigin(ballTexture.getSize().x * 0.5f, ballTexture.getSize().y * 0.5f);
				s.setPosition(ball._position);

				blueRT.draw(s);
			}

			if (ball._position.y + ballRadius < 0.0f) {
				sf::Sprite s;
				s.setTexture(arrowTexture);
				s.setOrigin(arrowTexture.getSize().x * 0.5f, 0.0f);
				s.setPosition(ball._position.x, 0.0f);

				blueRT.draw(s);
			}

			{
				sf::Text scoreText;
				scoreText.setFont(scoreFont);
				scoreText.setString(std::to_string(scoreBlue));
				scoreText.setCharacterSize(100);

				float width = scoreText.getLocalBounds().width;

				scoreText.setPosition(fieldCenter.x - width * 0.5f - 100.0f, 10.0f);

				scoreText.setColor(sf::Color(100, 133, 255));

				blueRT.draw(scoreText);
			}

			{
				sf::Text scoreText;
				scoreText.setFont(scoreFont);
				scoreText.setString(std::to_string(scoreRed));
				scoreText.setCharacterSize(100);

				float width = scoreText.getLocalBounds().width;

				scoreText.setPosition(fieldCenter.x - width * 0.5f + 100.0f, 10.0f);

				scoreText.setColor(sf::Color(255, 100, 100));

				blueRT.draw(scoreText);
			}

			blueRT.display();
		}

		skipFrameCounter = (skipFrameCounter + 1) % frameSkip;
	} while (!quit);

	return 0;
}

#endif
#include "Runner.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_RUNNER

void Runner::Limb::create(b2World* pWorld, const std::vector<LimbSegmentDesc> &descs, b2Body* pAttachBody, const b2Vec2 &localAttachPoint, uint16 categoryBits, uint16 maskBits) {
	_segments.resize(descs.size());

	b2Body* pPrevBody = pAttachBody;
	b2Vec2 prevAttachPoint = localAttachPoint;

	for (int si = 0; si < _segments.size(); si++) {
		b2BodyDef bodyDef;

		bodyDef.type = b2_dynamicBody;

		float offset = descs[si]._length * 0.5f - descs[si]._thickness * 0.5f;

		float angle = pPrevBody->GetAngle() + descs[si]._relativeAngle;

		bodyDef.position = pPrevBody->GetWorldPoint(prevAttachPoint) + b2Vec2(std::cos(angle) * offset, std::sin(angle) * offset);
		bodyDef.angle = angle;
		bodyDef.allowSleep = false;

		_segments[si]._pBody = pWorld->CreateBody(&bodyDef);

		_segments[si]._bodyShape.SetAsBox(descs[si]._length * 0.5f, descs[si]._thickness * 0.5f);

		b2FixtureDef fixtureDef;

		fixtureDef.shape = &_segments[si]._bodyShape;

		fixtureDef.density = descs[si]._density;

		fixtureDef.friction = descs[si]._friction;

		fixtureDef.restitution = descs[si]._restitution;

		fixtureDef.filter.categoryBits = categoryBits;
		fixtureDef.filter.maskBits = maskBits;

		_segments[si]._pBody->CreateFixture(&fixtureDef);

		b2RevoluteJointDef jointDef;

		jointDef.bodyA = pPrevBody;

		jointDef.bodyB = _segments[si]._pBody;

		jointDef.referenceAngle = descs[si]._relativeAngle;
		jointDef.localAnchorA = prevAttachPoint;
		jointDef.localAnchorB = b2Vec2(-offset, 0.0f);
		jointDef.collideConnected = false;
		jointDef.lowerAngle = descs[si]._minAngle;
		jointDef.upperAngle = descs[si]._maxAngle;
		jointDef.enableLimit = true;
		jointDef.maxMotorTorque = descs[si]._maxTorque;
		jointDef.motorSpeed = descs[si]._maxSpeed;
		jointDef.enableMotor = descs[si]._motorEnabled;

		_segments[si]._maxSpeed = descs[si]._maxSpeed;
		_segments[si]._minAngle = descs[si]._minAngle;
		_segments[si]._maxAngle = descs[si]._maxAngle;

		_segments[si]._pJoint = static_cast<b2RevoluteJoint*>(pWorld->CreateJoint(&jointDef));

		pPrevBody = _segments[si]._pBody;
		prevAttachPoint = b2Vec2(offset, 0.0f);
	}
}

void Runner::Limb::remove(b2World* pWorld) {
	for (int si = _segments.size() - 1; si >= 0; si--) {
		pWorld->DestroyJoint(_segments[si]._pJoint);
		pWorld->DestroyBody(_segments[si]._pBody);
	}
}

Runner::~Runner() {
	if (_world != nullptr) {
		_leftBackLimb.remove(_world.get());
		_leftFrontLimb.remove(_world.get());

		_rightBackLimb.remove(_world.get());
		_rightFrontLimb.remove(_world.get());

		_world->DestroyBody(_pBody);
	}
}

void Runner::createDefault(const std::shared_ptr<b2World> &world, const b2Vec2 &position, float angle, int layer) {
	const float bodyWidth = 0.45f;
	const float legInset = 0.075f;
	const float bodyHeight = 0.1f;
	const float bodyDensity = 1.0f;
	const float bodyFriction = 1.0f;
	const float bodyRestitution = 0.01f;

	std::vector<LimbSegmentDesc> leftSegments(3);

	leftSegments[0]._relativeAngle = 3.141596f * -0.25f;
	leftSegments[1]._relativeAngle = 3.141596f * -0.5f;
	leftSegments[2]._relativeAngle = 3.141596f * 0.5f;

	leftSegments[0]._length = 0.08f;
	leftSegments[1]._length = 0.13f;
	leftSegments[2]._length = 0.13f;

	std::vector<LimbSegmentDesc> rightSegments(2);

	rightSegments[0]._relativeAngle = 3.141596f * -0.75f;
	rightSegments[1]._relativeAngle = 3.141596f * 0.5f;

	rightSegments[0]._length = 0.15f;
	rightSegments[1]._length = 0.15f;

	_world = world;

	b2BodyDef bodyDef;

	bodyDef.type = b2_dynamicBody;

	bodyDef.position = position;
	bodyDef.angle = angle;
	bodyDef.allowSleep = false;

	_pBody = world->CreateBody(&bodyDef);

	_bodyShape.SetAsBox(bodyWidth * 0.5f, bodyHeight * 0.5f);

	b2FixtureDef fixtureDef;

	fixtureDef.shape = &_bodyShape;

	fixtureDef.density = bodyDensity;

	fixtureDef.friction = bodyFriction;

	fixtureDef.restitution = bodyRestitution;

	fixtureDef.filter.categoryBits = 1 << layer;
	fixtureDef.filter.maskBits = 1;

	_pBody->CreateFixture(&fixtureDef);

	_leftBackLimb.create(world.get(), leftSegments, _pBody, b2Vec2(-bodyWidth * 0.5f + legInset, -bodyHeight * 0.5f), 1 << layer, 1);
	_leftFrontLimb.create(world.get(), leftSegments, _pBody, b2Vec2(-bodyWidth * 0.5f + legInset, -bodyHeight * 0.5f), 1 << layer, 1);

	_rightBackLimb.create(world.get(), rightSegments, _pBody, b2Vec2(bodyWidth * 0.5f - legInset, -bodyHeight * 0.5f), 1 << (layer + 1), 1);
	_rightFrontLimb.create(world.get(), rightSegments, _pBody, b2Vec2(bodyWidth * 0.5f - legInset, -bodyHeight * 0.5f), 1 << (layer + 1), 1);
}

void Runner::renderDefault(sf::RenderTarget &rt, const sf::Color &color, float metersToPixels) {
	// Render back legs
	for (int si = _leftBackLimb._segments.size() - 1; si >= 0; si--) {
		int numVertices = _leftBackLimb._segments[si]._bodyShape.GetVertexCount();

		sf::ConvexShape shape;

		shape.setPointCount(numVertices);

		for (int i = 0; i < numVertices; i++)
			shape.setPoint(i, sf::Vector2f(_leftBackLimb._segments[si]._bodyShape.GetVertex(i).x, _leftBackLimb._segments[si]._bodyShape.GetVertex(i).y));

		shape.setPosition(metersToPixels * sf::Vector2f(_leftBackLimb._segments[si]._pBody->GetPosition().x, -_leftBackLimb._segments[si]._pBody->GetPosition().y));
		shape.setRotation(-_leftBackLimb._segments[si]._pBody->GetAngle() * 180.0f / 3.141596f);
		shape.setScale(metersToPixels, -metersToPixels);

		shape.setFillColor(mulColors(sf::Color(200, 200, 200), color));
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(0.01f);

		rt.draw(shape);
	}

	for (int si = _rightBackLimb._segments.size() - 1; si >= 0; si--) {
		int numVertices = _rightBackLimb._segments[si]._bodyShape.GetVertexCount();

		sf::ConvexShape shape;

		shape.setPointCount(numVertices);

		for (int i = 0; i < numVertices; i++)
			shape.setPoint(i, sf::Vector2f(_rightBackLimb._segments[si]._bodyShape.GetVertex(i).x, _rightBackLimb._segments[si]._bodyShape.GetVertex(i).y));

		shape.setPosition(metersToPixels * sf::Vector2f(_rightBackLimb._segments[si]._pBody->GetPosition().x, -_rightBackLimb._segments[si]._pBody->GetPosition().y));
		shape.setRotation(-_rightBackLimb._segments[si]._pBody->GetAngle() * 180.0f / 3.141596f);
		shape.setScale(metersToPixels, -metersToPixels);

		shape.setFillColor(mulColors(sf::Color(200, 200, 200), color));
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(0.01f);

		rt.draw(shape);
	}

	// Render body
	{
		int numVertices = _bodyShape.GetVertexCount();

		sf::ConvexShape shape;

		shape.setPointCount(numVertices);

		for (int i = 0; i < numVertices; i++)
			shape.setPoint(i, sf::Vector2f(_bodyShape.GetVertex(i).x, _bodyShape.GetVertex(i).y));

		shape.setPosition(metersToPixels * sf::Vector2f(_pBody->GetPosition().x, -_pBody->GetPosition().y));
		shape.setRotation(-_pBody->GetAngle() * 180.0f / 3.141596f);
		shape.setScale(metersToPixels, -metersToPixels);

		shape.setFillColor(mulColors(sf::Color::White, color));
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(0.01f);

		rt.draw(shape);
	}

	// Render front legs
	for (int si = 0; si < _leftFrontLimb._segments.size(); si++) {
		int numVertices = _leftFrontLimb._segments[si]._bodyShape.GetVertexCount();

		sf::ConvexShape shape;

		shape.setPointCount(numVertices);

		for (int i = 0; i < numVertices; i++)
			shape.setPoint(i, sf::Vector2f(_leftFrontLimb._segments[si]._bodyShape.GetVertex(i).x, _leftFrontLimb._segments[si]._bodyShape.GetVertex(i).y));

		shape.setPosition(metersToPixels * sf::Vector2f(_leftFrontLimb._segments[si]._pBody->GetPosition().x, -_leftFrontLimb._segments[si]._pBody->GetPosition().y));
		shape.setRotation(-_leftFrontLimb._segments[si]._pBody->GetAngle() * 180.0f / 3.141596f);
		shape.setScale(metersToPixels, -metersToPixels);

		shape.setFillColor(mulColors(sf::Color::White, color));
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(0.01f);

		rt.draw(shape);
	}

	for (int si = 0; si < _rightFrontLimb._segments.size(); si++) {
		int numVertices = _rightFrontLimb._segments[si]._bodyShape.GetVertexCount();

		sf::ConvexShape shape;

		shape.setPointCount(numVertices);

		for (int i = 0; i < numVertices; i++)
			shape.setPoint(i, sf::Vector2f(_rightFrontLimb._segments[si]._bodyShape.GetVertex(i).x, _rightFrontLimb._segments[si]._bodyShape.GetVertex(i).y));

		shape.setPosition(metersToPixels * sf::Vector2f(_rightFrontLimb._segments[si]._pBody->GetPosition().x, -_rightFrontLimb._segments[si]._pBody->GetPosition().y));
		shape.setRotation(-_rightFrontLimb._segments[si]._pBody->GetAngle() * 180.0f / 3.141596f);
		shape.setScale(metersToPixels, -metersToPixels);

		shape.setFillColor(mulColors(sf::Color::White, color));
		shape.setOutlineColor(sf::Color::Black);
		shape.setOutlineThickness(0.01f);

		rt.draw(shape);
	}
}

void Runner::getStateVector(std::vector<float> &state) {
	const int stateSize = 3 + 3 + 2 + 2 + 1 + 2 + 2;

	if (state.size() != stateSize)
		state.resize(stateSize);

	int si = 0;

	for (int i = 0; i < 3; i++)
		state[si++] = _leftBackLimb._segments[i]._pJoint->GetJointAngle();

	for (int i = 0; i < 3; i++)
		state[si++] = _leftFrontLimb._segments[i]._pJoint->GetJointAngle();

	for (int i = 0; i < 2; i++)
		state[si++] = _rightBackLimb._segments[i]._pJoint->GetJointAngle();

	for (int i = 0; i < 2; i++)
		state[si++] = _rightFrontLimb._segments[i]._pJoint->GetJointAngle();

	state[si++] = _pBody->GetAngle();

	b2ContactEdge* pEdge;

	state[si] = 0.0f;
	
	pEdge = _leftBackLimb._segments.back()._pBody->GetContactList();

	while (pEdge != nullptr) {
		if (pEdge->contact->IsTouching() && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
			state[si++] = 1.0f;

			break;
		}

		pEdge = pEdge->next;
	}

	state[si] = 0.0f;

	pEdge = _leftFrontLimb._segments.back()._pBody->GetContactList();

	while (pEdge != nullptr) {
		if (pEdge->contact->IsTouching() && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
			state[si++] = 1.0f;

			break;
		}

		pEdge = pEdge->next;
	}

	state[si] = 0.0f;

	pEdge = _rightBackLimb._segments.back()._pBody->GetContactList();

	while (pEdge != nullptr) {
		if (pEdge->contact->IsTouching() && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
			state[si++] = 1.0f;

			break;
		}

		pEdge = pEdge->next;
	}

	state[si] = 0.0f;

	pEdge = _rightFrontLimb._segments.back()._pBody->GetContactList();

	while (pEdge != nullptr) {
		if (pEdge->contact->IsTouching() && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && pEdge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
			state[si++] = 1.0f;

			break;
		}

		pEdge = pEdge->next;
	}
}

void Runner::motorUpdate(const std::vector<float> &action, float interpolateFactor) {
	int ai = 0;

	for (int i = 0; i < 3; i++) {
		float target = action[ai++] * (_leftBackLimb._segments[i]._maxAngle - _leftBackLimb._segments[i]._minAngle) + _leftBackLimb._segments[i]._minAngle;

		float speed = interpolateFactor * (target - _leftBackLimb._segments[i]._pJoint->GetJointAngle());

		//if (std::abs(speed) > _leftBackLimb._segments[i]._maxSpeed)
		//	speed = speed > 0.0f ? _leftBackLimb._segments[i]._maxSpeed : -_leftBackLimb._segments[i]._maxSpeed;

		_leftBackLimb._segments[i]._pJoint->SetMotorSpeed(speed);
	}

	for (int i = 0; i < 3; i++) {
		float target = action[ai++] * (_leftFrontLimb._segments[i]._maxAngle - _leftFrontLimb._segments[i]._minAngle) + _leftFrontLimb._segments[i]._minAngle;

		float speed = interpolateFactor * (target - _leftFrontLimb._segments[i]._pJoint->GetJointAngle());

		//if (std::abs(speed) > _leftFrontLimb._segments[i]._maxSpeed)
		//	speed = speed > 0.0f ? _leftFrontLimb._segments[i]._maxSpeed : -_leftFrontLimb._segments[i]._maxSpeed;

		_leftFrontLimb._segments[i]._pJoint->SetMotorSpeed(speed);
	}

	for (int i = 0; i < 2; i++) {
		float target = action[ai++] * (_rightBackLimb._segments[i]._maxAngle - _rightBackLimb._segments[i]._minAngle) + _rightBackLimb._segments[i]._minAngle;

		float speed = interpolateFactor * (target - _rightBackLimb._segments[i]._pJoint->GetJointAngle());

		//if (std::abs(speed) > _rightBackLimb._segments[i]._maxSpeed)
		//	speed = speed > 0.0f ? _rightBackLimb._segments[i]._maxSpeed : -_rightBackLimb._segments[i]._maxSpeed;

		_rightBackLimb._segments[i]._pJoint->SetMotorSpeed(speed);
	}

	for (int i = 0; i < 2; i++) {
		float target = action[ai++] * (_rightFrontLimb._segments[i]._maxAngle - _rightFrontLimb._segments[i]._minAngle) + _rightFrontLimb._segments[i]._minAngle;

		float speed = interpolateFactor * (target - _rightFrontLimb._segments[i]._pJoint->GetJointAngle());

		//if (std::abs(speed) > _rightFrontLimb._segments[i]._maxSpeed)
		//	speed = speed > 0.0f ? _rightFrontLimb._segments[i]._maxSpeed : -_rightFrontLimb._segments[i]._maxSpeed;

		_rightFrontLimb._segments[i]._pJoint->SetMotorSpeed(speed);
	}
}

#endif
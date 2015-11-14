#include "PrettySDR.h"

using namespace vis;

void PrettySDR::create(int width, int height) {
	_width = width;
	_height = height;

	_nodes.clear();
	_nodes.assign(width * height, 0.0f);
}

void PrettySDR::draw(sf::RenderTarget &rt, const sf::Vector2f &position) {
	float rWidth = _nodeSpaceSize * _width;
	float rHeight = _nodeSpaceSize * _height;

	sf::RectangleShape rsHorizontal;
	rsHorizontal.setPosition(position + sf::Vector2f(0.0f, _edgeRadius));
	rsHorizontal.setSize(sf::Vector2f(rWidth, rHeight - _edgeRadius * 2.0f));
	rsHorizontal.setFillColor(_backgroundColor);
	
	rt.draw(rsHorizontal);

	sf::RectangleShape rsVertical;
	rsVertical.setPosition(position + sf::Vector2f(_edgeRadius, 0.0f));
	rsVertical.setSize(sf::Vector2f(rWidth - _edgeRadius * 2.0f, rHeight));
	rsVertical.setFillColor(_backgroundColor);

	rt.draw(rsVertical);

	// Corners
	sf::CircleShape corner;
	corner.setRadius(_edgeRadius);
	corner.setPointCount(_edgeSegments);
	corner.setFillColor(_backgroundColor);
	corner.setOrigin(sf::Vector2f(_edgeRadius, _edgeRadius));

	corner.setPosition(position + sf::Vector2f(_edgeRadius, _edgeRadius));
	rt.draw(corner);

	corner.setPosition(position + sf::Vector2f(rWidth - _edgeRadius, _edgeRadius));
	rt.draw(corner);

	corner.setPosition(position + sf::Vector2f(rWidth - _edgeRadius, rHeight - _edgeRadius));
	rt.draw(corner);

	corner.setPosition(position + sf::Vector2f(_edgeRadius, rHeight - _edgeRadius));
	rt.draw(corner);

	// Nodes
	sf::CircleShape outer;
	outer.setRadius(_nodeSpaceSize * _nodeOuterRatio * 0.5f);
	outer.setPointCount(_nodeOuterSegments);
	outer.setFillColor(_nodeOuterColor);
	outer.setOrigin(sf::Vector2f(outer.getRadius(), outer.getRadius()));

	sf::CircleShape inner;
	inner.setRadius(_nodeSpaceSize * _nodeOuterRatio * _nodeInnerRatio * 0.5f);
	inner.setPointCount(_nodeInnerSegments);
	inner.setFillColor(_nodeInnerColor);
	inner.setOrigin(sf::Vector2f(inner.getRadius(), inner.getRadius()));

	for (int x = 0; x < _width; x++)
		for (int y = 0; y < _height; y++) {
			outer.setPosition(position + sf::Vector2f(x * _nodeSpaceSize + _edgeRadius, y * _nodeSpaceSize + _edgeRadius));

			rt.draw(outer);

			inner.setPosition(position + sf::Vector2f(x * _nodeSpaceSize + _edgeRadius, y * _nodeSpaceSize + _edgeRadius));

			inner.setFillColor(sf::Color(_nodeInnerColor.r, _nodeInnerColor.g, _nodeInnerColor.b, 255 * at(x, y)));

			rt.draw(inner);
		}
}
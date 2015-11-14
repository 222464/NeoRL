#pragma once

#include <SFML/Graphics.hpp>

namespace vis {
	class PrettySDR {
	private:
		std::vector<float> _nodes;

		int _width, _height;

	public:
		float _edgeRadius;
		float _nodeSpaceSize;
		float _nodeOuterRatio;
		float _nodeInnerRatio;
		
		int _edgeSegments;
		int _nodeOuterSegments;
		int _nodeInnerSegments;

		sf::Color _backgroundColor;
		sf::Color _nodeOuterColor;
		sf::Color _nodeInnerColor;

		PrettySDR()
			: _edgeRadius(4.0f), _nodeSpaceSize(16.0f), _nodeOuterRatio(0.85f), _nodeInnerRatio(0.75f),
			_edgeSegments(16), _nodeOuterSegments(16), _nodeInnerSegments(16),
			_backgroundColor(128, 128, 128), _nodeOuterColor(64, 64, 64), _nodeInnerColor(255, 0, 0)
		{}

		void create(int width, int height);

		float &operator[](int index) {
			return _nodes[index];
		}

		float &at(int x, int y) {
			return _nodes[x + y * _width];
		}

		void draw(sf::RenderTarget &rt, const sf::Vector2f &position);
	};
}
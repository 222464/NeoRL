#pragma once

// Inherit from this class to make the class uncopyable
namespace sys {
	class Uncopyable {
	protected:
		Uncopyable() {}
		virtual ~Uncopyable() {}
	private:
		Uncopyable(const Uncopyable &);
		Uncopyable &operator=(const Uncopyable &);
	};
}
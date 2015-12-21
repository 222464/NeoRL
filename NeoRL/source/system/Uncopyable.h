#pragma once

namespace sys {
	/*!
	\brief Inherit from this class to make it Uncopyable
	*/
	class Uncopyable {
	protected:
		Uncopyable() {}
		virtual ~Uncopyable() {}
	private:
		Uncopyable(const Uncopyable &);
		Uncopyable &operator=(const Uncopyable &);
	};
}
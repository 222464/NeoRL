#pragma once

#include <system/ComputeSystem.h>

#include <assert.h>

namespace sys {
	class ComputeProgram {
	private:
		cl::Program _program;

	public:
		bool loadFromFile(const std::string &name, ComputeSystem &cs);

		cl::Program &getProgram() {
			return _program;
		}
	};
}
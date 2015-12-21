#pragma once

#include <system/ComputeSystem.h>

#include <assert.h>

namespace sys {
	/*!
	\brief Compute program
	Holds OpenCL compute program with their associated kernels
	*/
	class ComputeProgram {
	private:
		/*!
		\brief OpenCL program
		*/
		cl::Program _program;

	public:
		/*!
		\brief Load from file
		Load program from a file
		*/
		bool loadFromFile(const std::string &name, ComputeSystem &cs);

		/*!
		\brief Get the underlying OpenCL program
		*/
		cl::Program &getProgram() {
			return _program;
		}
	};
}
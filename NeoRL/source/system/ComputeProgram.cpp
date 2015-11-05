#include "ComputeProgram.h"

#include <fstream>
#include <iostream>

using namespace sys;

bool ComputeProgram::loadFromFile(const std::string &name, ComputeSystem &cs) {
	std::ifstream fromFile(name);

	if (!fromFile.is_open()) {
#ifdef SYS_DEBUG
		std::cerr << "Could not open file " << name << "!" << std::endl;
#endif
		return false;
	}

	std::string source = "";

	while (!fromFile.eof() && fromFile.good()) {
		std::string line; 

		std::getline(fromFile, line);

		source += line + "\n";
	}

	_program = cl::Program(cs.getContext(), source);

	if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS) {
#ifdef SYS_DEBUG
		std::cerr << "Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << std::endl;
#endif
		return false;
	}

	return true;
}
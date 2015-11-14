#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

#include <random>
#include <assert.h>

namespace neo {
	enum BufferType {
		_front = 0, _back = 1
	};

	typedef std::array<cl::Image2D, 2> DoubleBuffer2D;
	typedef std::array<cl::Image3D, 2> DoubleBuffer3D;

	DoubleBuffer2D createDoubleBuffer2D(sys::ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType);
	DoubleBuffer3D createDoubleBuffer3D(sys::ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType);

	void randomUniform(cl::Image2D &image2D, sys::ComputeSystem &cs, cl::Kernel &randomUniform2DKernel, cl_int2 size, cl_float2 range, std::mt19937 &rng);
	void randomUniform(cl::Image3D &image3D, sys::ComputeSystem &cs, cl::Kernel &randomUniform3DKernel, cl_int3 size, cl_float2 range, std::mt19937 &rng);
}
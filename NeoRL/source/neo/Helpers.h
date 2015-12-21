#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

#include <random>
#include <assert.h>

namespace neo {
	/*!
	\brief Buffer types (can be used as indices)
	*/
	enum BufferType {
		_front = 0, _back = 1
	};

	//!@{
	/*!
	\brief Double buffer types
	*/
	typedef std::array<cl::Image2D, 2> DoubleBuffer2D;
	typedef std::array<cl::Image3D, 2> DoubleBuffer3D;
	//!@}

	//!@{
	/*!
	\brief Double buffer creation helpers
	*/
	DoubleBuffer2D createDoubleBuffer2D(sys::ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType);
	DoubleBuffer3D createDoubleBuffer3D(sys::ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType);
	//!@}

	//!@{
	/*!
	\brief Double buffer initialization helpers
	*/
	void randomUniform(cl::Image2D &image2D, sys::ComputeSystem &cs, cl::Kernel &randomUniform2DKernel, cl_int2 size, cl_float2 range, std::mt19937 &rng);
	void randomUniform(cl::Image3D &image3D, sys::ComputeSystem &cs, cl::Kernel &randomUniform3DKernel, cl_int3 size, cl_float2 range, std::mt19937 &rng);
	void randomUniformXY(cl::Image2D &image2D, sys::ComputeSystem &cs, cl::Kernel &randomUniform2DXYKernel, cl_int2 size, cl_float2 range, std::mt19937 &rng);
	void randomUniformXY(cl::Image3D &image3D, sys::ComputeSystem &cs, cl::Kernel &randomUniform3DXYKernel, cl_int3 size, cl_float2 range, std::mt19937 &rng);
	void randomUniformXZ(cl::Image2D &image2D, sys::ComputeSystem &cs, cl::Kernel &randomUniform2DXZKernel, cl_int2 size, cl_float2 range, std::mt19937 &rng);
	void randomUniformXZ(cl::Image3D &image3D, sys::ComputeSystem &cs, cl::Kernel &randomUniform3DXZKernel, cl_int3 size, cl_float2 range, std::mt19937 &rng);
	//!@}
}
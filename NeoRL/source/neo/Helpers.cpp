#include "Helpers.h"

using namespace neo;

DoubleBuffer2D neo::createDoubleBuffer2D(sys::ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType) {
	DoubleBuffer2D db;
	
	db[_front] = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y);
	db[_back] = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y);

	return db;
}

DoubleBuffer3D neo::createDoubleBuffer3D(sys::ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType) {
	DoubleBuffer3D db;

	db[_front] = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y, size.z);
	db[_back] = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y, size.z);

	return db;
}

void neo::randomUniform(cl::Image2D &image2D, sys::ComputeSystem &cs, cl::Kernel &randomUniform2DKernel, cl_int2 size, cl_float2 range, std::mt19937 &rng) {
	int argIndex = 0;

	std::uniform_int_distribution<int> seedDist;

	cl_uint2 seed = { seedDist(rng), seedDist(rng) };

	randomUniform2DKernel.setArg(argIndex++, image2D);
	randomUniform2DKernel.setArg(argIndex++, seed);
	randomUniform2DKernel.setArg(argIndex++, range);

	cs.getQueue().enqueueNDRangeKernel(randomUniform2DKernel, cl::NullRange, cl::NDRange(size.x, size.y));
}

void neo::randomUniform(cl::Image3D &image3D, sys::ComputeSystem &cs, cl::Kernel &randomUniform3DKernel, cl_int3 size, cl_float2 range, std::mt19937 &rng) {
	int argIndex = 0;

	std::uniform_int_distribution<int> seedDist;

	cl_uint2 seed = { seedDist(rng), seedDist(rng) };

	randomUniform3DKernel.setArg(argIndex++, image3D);
	randomUniform3DKernel.setArg(argIndex++, seed);
	randomUniform3DKernel.setArg(argIndex++, range);

	cs.getQueue().enqueueNDRangeKernel(randomUniform3DKernel, cl::NullRange, cl::NDRange(size.x, size.y, size.z));
}
#include "ImageWhitener.h"

using namespace neo;

void ImageWhitener::create(sys::ComputeSystem &cs, sys::ComputeProgram &program, cl_int2 imageSize, cl_int imageFormat, cl_int imageType) {
	_imageSize = imageSize;

	_result = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(imageFormat, imageType), imageSize.x, imageSize.y);

	_whitenKernel = cl::Kernel(program.getProgram(), "whiten");
}

void ImageWhitener::filter(sys::ComputeSystem &cs, const cl::Image2D &input, cl_int kernelRadius, cl_float intensity) {
	int argIndex = 0;

	_whitenKernel.setArg(argIndex++, input);
	_whitenKernel.setArg(argIndex++, _result);
	_whitenKernel.setArg(argIndex++, _imageSize);
	_whitenKernel.setArg(argIndex++, kernelRadius);
	_whitenKernel.setArg(argIndex++, intensity);

	cs.getQueue().enqueueNDRangeKernel(_whitenKernel, cl::NullRange, cl::NDRange(_imageSize.x, _imageSize.y));
}
#pragma once

#include "../system/ComputeSystem.h"
#include "../system/ComputeProgram.h"

namespace neo {
	/*!
	\brief Image whitener
	Applies local whitening transformation to input
	*/
	class ImageWhitener {
	private:
		/*!
		\brief Kernels
		*/
		cl::Kernel _whitenKernel;

		/*!
		\brief Resulting whitened image
		*/
		cl::Image2D _result;

		/*!
		\brief Size of the whitened image
		*/
		cl_int2 _imageSize;

	public:
		/*!
		\brief Create the image whitener
		Requires the image size and format.
		*/
		void create(sys::ComputeSystem &cs, sys::ComputeProgram &program, cl_int2 imageSize, cl_int imageFormat, cl_int imageType);

		/*!
		\brief Filter (whiten) an image with a kernel radius
		*/
		void filter(sys::ComputeSystem &cs, const cl::Image2D &input, cl_int kernelRadius, cl_float intensity = 1024.0f);

		/*!
		\brief Return filtered image result
		*/
		const cl::Image2D &getResult() const {
			return _result;
		}
	};
}
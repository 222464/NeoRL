#pragma once

#include <system/Uncopyable.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>

#define SYS_DEBUG

#define SYS_ALLOW_CL_GL_CONTEXT 0

namespace sys {
	/*!
	\brief Compute system
	Holds OpenCL platform, device, context, and command queue
	*/
	class ComputeSystem : private Uncopyable {
	public:
		enum DeviceType {
			_cpu, _gpu, _all, _none
		};

	private:
		//!@{
		/*!
		\brief OpenCL handles
		*/
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;
		//!@}

	public:
		/*!
		\brief Create compute system with a given device type
		Optional: Create from an OpenGL context
		*/
		bool create(DeviceType type, bool createFromGLContext = false);

		/*!
		\brief Get underlying OpenCL platform
		*/
		cl::Platform &getPlatform() {
			return _platform;
		}

		/*!
		\brief Get underlying OpenCL device
		*/
		cl::Device &getDevice() {
			return _device;
		}

		/*!
		\brief Get underlying OpenCL context
		*/
		cl::Context &getContext() {
			return _context;
		}

		/*!
		\brief Get underlying OpenCL command queue
		*/
		cl::CommandQueue &getQueue() {
			return _queue;
		}
	};
}
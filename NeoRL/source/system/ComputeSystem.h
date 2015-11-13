#pragma once

#include <system/Uncopyable.h>

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>

#define SYS_DEBUG

#define SYS_ALLOW_CL_GL_CONTEXT 0

namespace sys {
	class ComputeSystem : private Uncopyable {
	public:
		enum DeviceType {
			_cpu, _gpu, _all, _none
		};

	private:
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;

	public:
		bool create(DeviceType type, bool createFromGLContext = false);

		cl::Platform &getPlatform() {
			return _platform;
		}

		cl::Device &getDevice() {
			return _device;
		}

		cl::Context &getContext() {
			return _context;
		}

		cl::CommandQueue &getQueue() {
			return _queue;
		}
	};
}
[![DOI](https://zenodo.org/badge/19602/222464/NeoRL.svg)](https://zenodo.org/badge/latestdoi/19602/222464/NeoRL)

![NeoRL Logo](http://i1218.photobucket.com/albums/dd401/222464/NeoRL_logo_med.png)

Welcome to NeoRL, an algorithmic GPU neocortex simulation library.

# Installation

NeoRL requires OpenCL 2.0 or greater to run. Unfortunately this excludes Nvidia hardware, but it will work for AMD and Intel processors.
If you have an AMD card, I recommend getting the AMD APP SDK here: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
Otherwise, you can try the Intel OpenCL SDK: https://software.intel.com/en-us/intel-opencl
It should also be feasible to run it on a Xeon Phi.

In order to run the demos, you will also need to install SFML. Additionally, one demo (Runner) requires Box2D in order to work.

To select a demo, change the macros in Settings.h.

You can generate documentation using Doxygen and the provided doxygen_config file.

# Overview

See the accompanying blog posts at to discover how NeoRL works internally at http://www.twistedkeyboardsoftware.com/
You don't need to know this in order to use it, it can be treated as a black box sequence predictor as well.

NeoRL contains both reinforcement learning agents as well as a predictive hierarchy. These are fully online learning algorithms.
The simplest usage of both the reinforcement learning agents and the predictive hierarchy involves calling:

```cpp
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Predictive Hierarchy ---------------------------

	// Temporary input buffer
	cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 2, 2);

	// Layer descriptors for hierarchy
	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 16, 16 };
	layerDescs[2]._size = { 16, 16 };

	// Hierarchy itself
	neo::PredictiveHierarchy ph;

	// 2x2 input field
	ph.createRandom(cs, prog, { 2, 2 }, layerDescs, { -0.01f, 0.01f }, 0.0f, generator);
```

You can then step the simulation with:

```cpp
	// Copy vals into temporary OpenCL image buffer
	cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 2, 2, 1 }, 0, 0, vals.data());

	// Step the simulation
	ph.simStep(cs, inputImage);

	// Retrieve the prediction (same dimensions as input field)
	std::vector<float> pred(4);

	cs.getQueue().enqueueReadImage(ph.getFirstLayerPred().getHiddenStates()[neo::_back], CL_TRUE, { 0, 0, 0 }, { 2, 2, 1 }, 0, 0, pred.data());
```

See the demos for more complicated usage.

# License

ZLib license. See LICENSE.md
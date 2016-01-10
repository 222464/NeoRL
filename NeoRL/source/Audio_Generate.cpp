#include <Settings.h>

#if EXPERIMENT_SELECTION == EXPERIMENT_AUDIO_GENERATE

#include <SFML/Audio.hpp>

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <neo/PredictiveHierarchy.h>

#include <complex>
#include <valarray>

#define RECONSTRUCT_DIRECT 1

const float pi = 3.14159265359;

typedef std::complex<float> Complex;
typedef std::valarray<Complex> CArray;

// Cooley�Tukey FFT (in-place)
void fft(CArray& x) {
	const size_t N = x.size();

	if (N <= 1)
		return;

	// Divide
	CArray even = x[std::slice(0, N / 2, 2)];
	CArray  odd = x[std::slice(1, N / 2, 2)];

	// Conquer
	fft(even);
	fft(odd);

	// Combine
	for (size_t k = 0; k < N / 2; ++k) {
		Complex t = std::polar(1.0f, -2.0f * pi * k / N) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}

// Inverse fft (in-place)
void ifft(CArray& x) {
	// Conjugate the complex numbers
	x = x.apply(std::conj);

	// Forward fft
	fft(x);

	// Conjugate the complex numbers again
	x = x.apply(std::conj);

	// Scale the numbers
	x /= x.size();
}

const int aeSamplesSize = 4096;
const float sampleScalar = 1.0f / std::pow(2.0f, 15.0f);
const float sampleScalarInv = 1.0f / sampleScalar;
const int reconStride = 2048;
const float sampleCurvePower = 1.0f;
const float sampleCurvePowerInv = 1.0f / sampleCurvePower;
const int trainStride = 2048;

float compress(float x) {
	return x * sampleScalar;// (x > 0 ? 1 : -1) * std::pow(std::abs(x) * sampleScalar, sampleCurvePower);
}

float decompress(float x) {
	return x * sampleScalarInv;// (x > 0 ? 1 : -1) * std::pow(std::abs(x), sampleCurvePowerInv) * sampleScalarInv;
}

struct FeaturesPointer {
	int _bufferStart;

	std::vector<float> _features;

	float similarity(const std::vector<float> &otherFeatures) {
		float sum = 0.0f;

		for (int i = 0; i < _features.size(); i++) {
			sum += _features[i] * otherFeatures[i];
		}

		return sum;
	}
};

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels.cl", cs);

	// --------------------------- Create the Sparse Coder ---------------------------

	const int dimV = std::sqrt(static_cast<float>(aeSamplesSize));
	
	cl::Image2D input = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), dimV, dimV);
	std::vector<float> visibleStates(dimV * dimV, 0.0f);
	CArray fftBuffer(visibleStates.size());
	std::vector<float> pred(visibleStates.size(), 0.0f);

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 32, 32 };
	layerDescs[1]._size = { 32, 32 };
	layerDescs[2]._size = { 32, 32 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { dimV, dimV }, layerDescs, { -0.01f, 0.01f }, generator);

	ph._whiteningKernelRadius = 2;

	sf::SoundBuffer buffer;

	buffer.loadFromFile("testSound4.wav");

	std::cout << "Training on sound..." << std::endl;

	int featuresCount = static_cast<int>(std::floor(buffer.getSampleCount() / static_cast<float>(trainStride)));
	
	for (int t = 0; t < 15; t++) {
		for (int s = 0; s < featuresCount; s++) {
			// Extract features
			int start = s * trainStride;

			for (int i = 0; i < aeSamplesSize; i++) {
				int si = start + i;

				if (si < buffer.getSampleCount())
					fftBuffer[i] = compress(buffer.getSamples()[si]);
				else
					fftBuffer[i] = 0.0f;
			}

			//fft(fftBuffer);

			for (int i = 0; i < aeSamplesSize; i++)
				visibleStates[i] = fftBuffer[i].real();
			
			cs.getQueue().enqueueWriteImage(input, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(dimV), static_cast<cl::size_type>(dimV), 1 }, 0, 0, visibleStates.data());

			ph.simStep(cs, input);

			cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { static_cast<cl::size_type>(dimV), static_cast<cl::size_type>(dimV), 1 }, 0, 0, pred.data());

			if (s % 10 == 0) {
				std::cout << "Sample: " << s << "/" << featuresCount << std::endl;
			}
		}

		std::cout << "Pass " << t << std::endl;
	}

	std::cout << "Generating extra..." << std::endl;

	// Extend song
	int extraFeatures = 1200;

	std::vector<float> extraSamplesf((extraFeatures + 1) * trainStride, 0.0f);
	std::vector<float> extraSamplesSums((extraFeatures + 1) * trainStride, 0.0f);
	std::normal_distribution<float> noiseDist(0.0f, 1.0f);

	for (int s = 0; s < extraFeatures; s++) {
		if (s < featuresCount) {
			int start = s * trainStride;

			for (int i = 0; i < aeSamplesSize; i++) {
				int si = start + i;

				if (si < buffer.getSampleCount())
					fftBuffer[i] = compress(buffer.getSamples()[si]);
				else
					fftBuffer[i] = 0.0f;
			}

			//fft(fftBuffer);

			for (int i = 0; i < aeSamplesSize; i++)
				visibleStates[i] = fftBuffer[i].real();

			cs.getQueue().enqueueWriteImage(input, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(dimV), static_cast<cl::size_type>(dimV), 1 }, 0, 0, visibleStates.data());
		}
		else {
			for (int i = 0; i < aeSamplesSize; i++)
				visibleStates[i] = std::min(1.0f, std::max(-1.0f, std::min(1.0f, std::max(-1.0f, pred[i])) + noiseDist(generator) * 0.3f));

			cs.getQueue().enqueueWriteImage(input, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(dimV), static_cast<cl::size_type>(dimV), 1 }, 0, 0, visibleStates.data());
		}

		ph.simStep(cs, input, false);

		cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(dimV), static_cast<cl::size_type>(dimV), 1 }, 0, 0, pred.data());

		for (int i = 0; i < aeSamplesSize; i++)
			fftBuffer[i] = Complex(pred[i], 0.0f);

		//ifft(fftBuffer);

		int start = s * trainStride;

		for (int i = 0; i < aeSamplesSize; i++) {
			extraSamplesf[start + i] += fftBuffer[i].real();
			extraSamplesSums[start + i]++;
		}

		if (s % 10 == 0) {
			std::cout << "Sample: " << s << "/" << extraFeatures << std::endl;
		}
	}

	std::vector<sf::Int16> extraSamples(extraSamplesf.size());

	for (int i = 0; i < extraSamplesf.size(); i++)
		if (extraSamplesSums[i] != 0)
			extraSamples[i] = static_cast<sf::Int16>(decompress(extraSamplesf[i] / extraSamplesSums[i]));

	sf::SoundBuffer extraBuffer;

	extraBuffer.loadFromSamples(extraSamples.data(), extraSamples.size(), 1, buffer.getSampleRate());

	std::cout << "Saving extra sound..." << std::endl;

	extraBuffer.saveToFile("extra.wav");

	std::cout << "Done." << std::endl;

	return 0;
}

#endif
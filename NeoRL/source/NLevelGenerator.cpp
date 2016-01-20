#include <Settings.h>

#if EXPERIMENT_SELECTION == EXPERIMENT_N_LEVEL_GENERATOR

#include <fstream>
#include <sstream>

#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <neo/PredictiveHierarchy.h>

enum Section {
	_name = 0, _author, _type, _data, _end
};

struct Level {
	std::string _name;
	std::string _author;
	std::string _type;

	std::string _tileData;
	std::string _objectData;
};

void loadLevels(const std::string &fileName, std::vector<Level> &levels, std::unordered_set<char> &tileDataCharset, std::unordered_set<char> &objectDataCharset) {
	std::ifstream fromFile(fileName);

	// Skip first 50 lines
	for (int i = 0; i < 50; i++) {
		std::string line;

		std::getline(fromFile, line);
	}

	while (!fromFile.eof() && fromFile.good()) {
		std::string line;

		std::getline(fromFile, line);

		// If line starts with "$", then it is a level
		if (line[0] == '$') {
			Level l;

			int lastIndex = 1;

			for (int i = 0; i < _end; i++) {
				// Search for next delimiter
				int d;

				for (d = lastIndex; d < line.length(); d++)
					if (line[d] == '#')
						break;

				std::string data = line.substr(lastIndex, d - lastIndex);		

				switch (i) {
				case _name:
					l._name = data;

					break;

				case _author:
					l._author = data;

					break;

				case _type:
					l._type = data;

					break;

				case _data:

					// Search for pipe, which splits data into 2 sections
					int k;

					for (k = 0; k < data.length(); k++)
						if (data[k] == '|')
							break;

					l._tileData = data.substr(0, k);
					l._objectData = data.substr(k + 1, data.length() - (k + 1));

					// Check charsets
					for (int t = 0; t < l._tileData.length(); t++) {
						if (tileDataCharset.find(l._tileData[t]) == tileDataCharset.end())
							tileDataCharset.insert(l._tileData[t]);
					}

					for (int t = 0; t < l._objectData.length(); t++) {
						if (objectDataCharset.find(l._objectData[t]) == objectDataCharset.end())
							objectDataCharset.insert(l._objectData[t]);
					}

					break;
				}

				lastIndex = d + 1;
			}

			levels.push_back(l);
		}
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	sys::ComputeSystem cs;

	cs.create(sys::ComputeSystem::_gpu);

	sys::ComputeProgram prog;

	prog.loadFromFile("resources/neoKernels2.cl", cs);

	std::vector<Level> levels;

	std::unordered_set<char> tileDataCharset;
	std::unordered_set<char> objectDataCharset;

	loadLevels("userlevels.txt", levels, tileDataCharset, objectDataCharset);

	// --------------------------- Create the Sparse Coder ---------------------------

	const int numTiles = levels.front()._tileData.length();

	const int charsetSize = 128 + 2; // + 2 indicating the portion it is on (01 for tiles, 10 for objects)

	const cl::size_type visDim = std::ceil(std::sqrt(static_cast<float>(charsetSize)));
	const int visArea = visDim * visDim;

	const int maxObjects = 1000;

	cl::Image2D input = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), visDim, visDim);

	std::vector<float> inputData(visArea, 0.0f);
	std::vector<float> predData(visArea, 0.0f);

	std::vector<neo::PredictiveHierarchy::LayerDesc> layerDescs(3);

	layerDescs[0]._size = { 16, 16 };
	layerDescs[1]._size = { 16, 16 };
	layerDescs[2]._size = { 16, 16 };

	neo::PredictiveHierarchy ph;

	ph.createRandom(cs, prog, { static_cast<int>(visDim), static_cast<int>(visDim) }, layerDescs, { -0.01f, 0.01f }, generator);

	ph._whiteningKernelRadius = 1;

	// Learn levels
	std::uniform_int_distribution<int> levelDist(0, levels.size() - 1);

	for (int iter = 0; iter < 20; iter++) {
		// Choose random level
		int index = levelDist(generator);

		const Level &l = levels[index];

		// Set mode for tiles
		inputData[charsetSize - 2] = 0.0f;
		inputData[charsetSize - 1] = 1.0f;

		// Run through once to get PH ready
		inputData['#'] = 1.0f;

		cs.getQueue().enqueueWriteImage(input, CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, inputData.data());

		ph.simStep(cs, input);

		inputData['#'] = 0.0f;

		// Run through tile data
		for (int i = 0; i < l._tileData.length(); i++) {
			// Set character to 1
			inputData[l._tileData[i]] = 1.0f;

			cs.getQueue().enqueueWriteImage(input, CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, inputData.data());

			ph.simStep(cs, input);

			// Unset character
			inputData[l._tileData[i]] = 0.0f;
		}

		// Set mode for objects
		inputData[charsetSize - 2] = 1.0f;
		inputData[charsetSize - 1] = 0.0f;

		// Run through once to get PH ready
		inputData['#'] = 1.0f;

		cs.getQueue().enqueueWriteImage(input, CL_TRUE, { 0, 0, 0 }, { visDim, visDim, 1 }, 0, 0, inputData.data());

		ph.simStep(cs, input);

		inputData['#'] = 0.0f;

		// Run through object data
		for (int i = 0; i < l._objectData.length(); i++) {
			// Set character to 1
			inputData[l._objectData[i]] = 1.0f;

			cs.getQueue().enqueueWriteImage(input, CL_TRUE, { 0, 0, 0 }, { visDim, visDim, 1 }, 0, 0, inputData.data());

			ph.simStep(cs, input);

			// Unset character
			inputData[l._objectData[i]] = 0.0f;
		}

		std::cout << "Went over level #" << (index + 1) << " \"" << l._name << "\"" << std::endl;
	}

	// Generate new maps
	std::ofstream toFile("generatedNLevels.txt");

	std::normal_distribution<float> noiseDist(0.0f, 1.0f);

	for (int i = 0; i < 10; i++) {
		toFile << "$" << "Generated Level " << (i + 1) << "#NeoRL#Experimental#";

		// Generated level data


		// Set mode for tiles
		inputData[charsetSize - 2] = 0.0f;
		inputData[charsetSize - 1] = 1.0f;

		// Run through once to get PH ready
		//cs.getQueue().enqueueWriteImage(input, CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, inputData.data());

		//ph.simStep(cs, input);

		char prevChar = 0;

		for (int i = 0; i < numTiles; i++) {
			// Set character to 1
			inputData[prevChar] = 1.0f;

			cs.getQueue().enqueueWriteImage(input, CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, inputData.data());

			ph.simStep(cs, input, false);

			// Unset character
			inputData[prevChar] = 0.0f;

			char newChar = 0;

			cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, predData.data());
	
			for (int j = 1; j < charsetSize - 2; j++)
				if (predData[j] > predData[newChar])
					newChar = j;

			// Add new character
			toFile << newChar;

			prevChar = newChar;
		}

		toFile << "|";

		// Set mode for objects
		inputData[charsetSize - 2] = 1.0f;
		inputData[charsetSize - 1] = 0.0f;

		// Run through once to get PH ready
		//cs.getQueue().enqueueWriteImage(input, CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, inputData.data());

		//ph.simStep(cs, input);

		prevChar = 0;

		for (int i = 0; i < maxObjects; i++) {
			// Set character to 1
			inputData[prevChar] = 1.0f;

			std::vector<float> noisyInputData = inputData;

			for (int j = 0; j < noisyInputData.size(); j++) {
				noisyInputData[j] += noiseDist(generator) * 0.1f;
			}

			cs.getQueue().enqueueWriteImage(input, CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, inputData.data());

			ph.simStep(cs, input, false);

			// Unset character
			inputData[prevChar] = 0.0f;

			char newChar = 0;

			cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, cl::array<cl::size_type, 3> { 0, 0, 0 }, cl::array<cl::size_type, 3> { visDim, visDim, 1 }, 0, 0, predData.data());

			for (int j = 1; j < charsetSize - 2; j++)
				if (predData[j] > predData[newChar])
					newChar = j;

			// If is delimiter, break
			if (newChar == '#')
				break;

			// Add new character
			toFile << newChar;

			prevChar = newChar;
		}

		toFile << "#" << std::endl << std::endl;
	}

	return 0;
}

#endif
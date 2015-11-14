#include "Settings.h"

#if EXPERIMENT_SELECTION == EXPERIMENT_RNN_BENCHMARK

#include <sdr/IPredictiveRSDR.h>

#include <simtree/SDRST.h>

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

int main() {
	std::mt19937 generator(time(nullptr));

	std::vector<std::vector<double>> timeSeries;

	std::ifstream fromFile("resources/data.txt");

	if (!fromFile.is_open()) {
		std::cerr << "Could not open data.txt!" << std::endl;

		return 1;
	}

	// Skip first line
	std::string line;

	std::getline(fromFile, line);

	int numEntries = 1;

	for (int i = 0; i < line.size(); i++)
		if (line[i] == ',')
			numEntries++;

	int numSkipEntries = 1;

	int numEntriesUse = numEntries - numSkipEntries;

	std::vector<double> minimums(numEntriesUse, 999999999.0);
	std::vector<double> maximums(numEntriesUse, -999999999.0);

	while (fromFile.good() && !fromFile.eof()) {
		std::vector<double> entries(numEntriesUse);

		std::string line;

		std::getline(fromFile, line);

		std::istringstream fromLine(line);

		std::string param;

		// Skip entries
		for (int i = 0; i < numSkipEntries; i++) {
			std::string entry;

			std::getline(fromLine, entry, ',');
		}

		for (int i = 0; i < numEntriesUse; i++) {
			std::string entry;

			std::getline(fromLine, entry, ',');

			if (entry == "")
				entries[i] = 0.0;
			else {
				double value = std::stod(entry);

				maximums[i] = std::max(maximums[i], value);
				minimums[i] = std::min(minimums[i], value);

				entries[i] = value;
			}
		}

		timeSeries.push_back(entries);
	}

	// Rescale
	for (int i = 0; i < timeSeries.size(); i++) {
		for (int j = 0; j < timeSeries[i].size(); j++) {
			timeSeries[i][j] = (timeSeries[i][j] - minimums[j]) / std::max(0.0001, (maximums[j] - minimums[j]));
		}
	}

	/*timeSeries.clear();

	timeSeries.resize(10);
	timeSeries[0] = { 0.0f, 1.0f, 0.0f };
	timeSeries[1] = { 0.0f, 0.0f, 0.0f };
	timeSeries[2] = { 1.0f, 1.0f, 0.0f };
	timeSeries[3] = { 0.0f, 0.0f, 1.0f };
	timeSeries[4] = { 0.0f, 1.0f, 0.0f };
	timeSeries[5] = { 0.0f, 0.0f, 1.0f };
	timeSeries[6] = { 0.0f, 0.0f, 0.0f };
	timeSeries[7] = { 0.0f, 0.0f, 0.0f };
	timeSeries[8] = { 0.0f, 1.0f, 0.0f };
	timeSeries[9] = { 0.0f, 1.0f, 1.0f };*/

	std::vector<sdr::IPredictiveRSDR::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 8;
	layerDescs[0]._height = 8;

	layerDescs[1]._width = 6;
	layerDescs[1]._height = 6;

	layerDescs[2]._width = 4;
	layerDescs[2]._height = 4;

	sdr::IPredictiveRSDR prsdr;

	prsdr.createRandom(4, 5, 8, layerDescs, -0.01f, 0.01f, 0.0f, generator);

	float avgError = 1.0f;

	float avgErrorDecay = 0.01f;

	for (int iter = 0; iter < 1000; iter++) {
		for (int i = 0; i < timeSeries.size(); i++) {
			float error = 0.0f;

			for (int j = 0; j < timeSeries[i].size(); j++) {
				error += std::pow(prsdr.getPrediction(j) - timeSeries[i][j], 2);

				prsdr.setInput(j, timeSeries[i][j]);
			}

			avgError = (1.0f - avgErrorDecay) * avgError + avgErrorDecay * error;

			prsdr.simStep(generator);

			if (i % 10 == 0) {
				std::cout << "Iteration " << i << ": " << avgError << std::endl;
			}
		}
	}

	return 0;
}

#endif
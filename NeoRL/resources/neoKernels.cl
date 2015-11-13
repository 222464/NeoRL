// ----------------------------------------- Samplers -----------------------------------------

constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

// ----------------------------------------- Common -----------------------------------------

float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

float randNormal(uint2* state) {
	float u1 = randFloat(state);
	float u2 = randFloat(state);

	return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

bool inBounds(int2 position, int2 upperBound) {
	return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
}

bool inBounds(int2 position, int2 lowerBound, int2 upperBound) {
	return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.x && position.y < upperBound.y;
}

float thresholdState(float state, float threshold) {
	return fmaxf(0.0f, fabsf(state) - threshold) * (state > 0.0f ? 1.0f : 0.0f);
}

// Initialize a random uniform 2D image
void kernel randomUniform2D(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;

	write_imagef(values, position, (float4)(weight));
}

// Initialize a random uniform 3D image
void kernel randomUniform3D(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76, get_global_id(1) * 21 + 42) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;

	write_imagef(values, position, (float4)(weight));
}

// ----------------------------------------- Sparse Coder -----------------------------------------

void kernel scReconstructVisibleError(read_only image2d_t hiddenStates, read_only image2d_t visibleStates,
	write_only image2d_t reconstructionError, read_only image3d_t weights,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x, visiblePosition.y * visibleToHidden.y);
	
	float recon = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
	for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
		int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
		if (hiddenPosition.x >= 0 && hiddenPosition.x < hiddenSize.x && hiddenPosition.y >= 0 && hiddenPosition.y < hiddenSize.y) {
			// Next layer node's receptive field
			int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x, hiddenPosition.y * hiddenToVisible.y);

			int2 fieldLowerBound = fieldCenter - (int2)(radius);
			int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
			// Check for containment
			if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {	
				int2 offset = visiblePosition - fieldLowerBound;

				float hiddenState = read_imagef(hiddenStates, hiddenPosition).x;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
				recon += hiddenState * weight;
			}
		}
	}

	float state = read_imagef(visibleStates, visiblePosition).x;

	float error = state - recon;

	write_imagef(reconstructionError, inputPosition, (float4)(error));
}

void kernel scActivateFromReconstructionError(read_only image2d_t reconstructionError,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * _hiddenToVisible.x, hiddenPosition.y * _hiddenToVisible.y);
	
	float sum = read_imagef(hiddenSummationTempBack, hiddenPosition).x;

	int wi = 0;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds(visiblePosition, visibleSize)) {
				float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0));

				float error = read_imagef(reconstructionError, visiblePosition).x;

				sum += weight * error;
			}

			wi++;
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum));
}

void kernel scSolveHidden(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront, read_only image2d_t hiddenThresholds, float stepSize) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = read_imagef(hiddenSummationTemp, hiddenPosition).x;

	float statePrev = read_imagef(hiddenStatesBack, hiddenPosition).x;

	float threshold = read_imagef(hiddenThresholds, hiddenPosition).x;

	float state = thresholdState(statePrev + sum * stepSize, threshold);

	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state));
}

void kernel scLearnThresholds(read_only image2d_t hiddenThresholdsBack, write_only image2d_t hiddenThresholdsFront,
	read_only image2d_t hiddenStates,
	float thresholdAlpha, float activeRatio)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thresholdPrev = read_imagef(hiddenThresholdsBack, hiddenPosition).x;

	float hiddenState = read_imagef(hiddenStates, hiddenPosition).x;

	float threshold = fmaxf(0.0f, thresholdPrev + thresholdAlpha * (hiddenState - activeRatio));

	write_imagef(hiddenThresholdsFront, hiddenPosition, (float4)(threshold));
}

void kernel scLearnSparseCoderWeights(read_only image2d_t reconstructionError,
	read_only image2d_t hiddenStates, read_only image3d_t weightsBack, read_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * _hiddenToVisible.x, hiddenPosition.y * _hiddenToVisible.y);

	float state = read_imagef(hiddenStates, hiddenPosition).x;

	int wi = 0;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds(visiblePosition, visibleSize)) {
				float weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0));

				float error = read_imagef(reconstructionError, visiblePosition).x;

				float weight = weightPrev + weightAlpha * state * error;

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight));
			}

			wi++;
		}
}
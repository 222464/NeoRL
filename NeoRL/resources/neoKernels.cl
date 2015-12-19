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

constant float minFloatEpsilon = 0.0001f;

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

float relu(float x, float leak) {
	return x > 0.0f ? x : x * leak;
}

float relud(float x, float leak) {
	return x > 0.0f ? 1.0f : leak;
}

float elu(float x, float alpha) {
	return x >= 0.0f ? x : alpha * (exp(x) - 1.0f);
}

float elud(float x, float alpha) {
	return x >= 0.0f ? 1.0f : x + alpha;
}

bool inBounds0(int2 position, int2 upperBound) {
	return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
}

bool inBounds(int2 position, int2 lowerBound, int2 upperBound) {
	return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
}

// Initialize a random uniform 2D image (X field)
void kernel randomUniform2D(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;

	write_imagef(values, position, (float4)(value, 0.0f, 0.0f, 0.0f));
}

// Initialize a random uniform 3D image (X field)
void kernel randomUniform3D(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;

	write_imagef(values, (int4)(position, 0), (float4)(value, 0.0f, 0.0f, 0.0f));
}

// Initialize a random uniform 2D image (XY fields)
void kernel randomUniform2DXY(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 15 + 66, get_global_id(1) * 61 + 2) * 56;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, position, (float4)(v.x, v.y, 0.0f, 0.0f));
}

// Initialize a random uniform 2D image (XZ fields)
void kernel randomUniform2DXZ(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, position, (float4)(v.x, 0.0f, v.y, 0.0f));
}

// Initialize a random uniform 3D image (XY fields)
void kernel randomUniform3DXY(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, (int4)(position, 0), (float4)(v.x, v.y, 0.0f, 0.0f));
}

// Initialize a random uniform 3D image (XZ fields)
void kernel randomUniform3DXZ(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, (int4)(position, 0), (float4)(v.x, 0.0f, v.y, 0.0f));
}

// ----------------------------------------- Comparison Sparse Coder -----------------------------------------

void kernel cscForwardError(read_only image2d_t hiddenStates, read_only image2d_t visibleStates,
	write_only image2d_t reconstructionError, read_only image3d_t weights,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float recon = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

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

	write_imagef(reconstructionError, visiblePosition, (float4)(error));
}

void kernel cscActivate(read_only image2d_t visibleStates,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float sum = read_imagef(hiddenSummationTempBack, hiddenPosition).x;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float state = read_imagef(visibleStates, visiblePosition).x;

				sum += state * weight;
			}
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum));
}

void kernel cscActivateIgnoreMiddle(read_only image2d_t visibleStates,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float sum = read_imagef(hiddenSummationTempBack, hiddenPosition).x;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			if (dx == 0 && dy == 0)
				continue;

			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float state = read_imagef(visibleStates, visiblePosition).x;

				sum += state * weight;
			}
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum));
}

void kernel cscSolveHidden(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront,
	int2 hiddenSize, int radius, float activeRatio)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float activation = read_imagef(hiddenSummationTemp, hiddenPosition).x;

	float statePrev = read_imagef(hiddenStatesBack, hiddenPosition).x;

	int2 fieldLowerBound = hiddenPosition - (int2)(radius);

	float inhibition = 0.0f;

	float counter = 0.0f;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			if (dx == 0 && dy == 0)
				continue;
			
			int2 otherPosition = hiddenPosition + (int2)(dx, dy);

			if (inBounds0(otherPosition, hiddenSize)) {
				float otherActivation = read_imagef(hiddenSummationTemp, otherPosition).x;

				inhibition += otherActivation >= activation ? 1.0f : 0.0f;

				counter++;
			}
		}

	float state = inhibition < (counter * activeRatio) ? 1.0f : 0.0f;

	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state));
}

void kernel cscLearnHiddenBiases(read_only image2d_t visibleBiasesBack, write_only image2d_t visibleBiasesFront,
	read_only image2d_t hiddenErrors, read_only image2d_t hiddenStates,
	float boostAlpha, float activeRatio)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float biasPrev = read_imagef(visibleBiasesBack, hiddenPosition).x;

	float state = read_imagef(hiddenStates, hiddenPosition).x;
	
	float bias = biasPrev + boostAlpha * (activeRatio - state);

	write_imagef(visibleBiasesFront, hiddenPosition, (float4)(bias));
}

void kernel cscLearnHiddenWeights(read_only image2d_t visibleErrors, read_only image2d_t visibleStates,
	read_only image2d_t hiddenErrors, read_only image2d_t hiddenStates,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	float state = read_imagef(hiddenStates, hiddenPosition).x;
	
	float error = read_imagef(hiddenErrors, hiddenPosition).x * state;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float visibleError = read_imagef(visibleErrors, visiblePosition).x;
				float visibleState = read_imagef(visibleStates, visiblePosition).x;

				float weight = weightPrev + weightAlpha * ((visibleError - weightPrev) * state);

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight));
			}
		}
}

void kernel cscLearnHiddenWeightsTraces(read_only image2d_t rewards, read_only image2d_t visibleErrors, read_only image2d_t visibleStates,
	read_only image2d_t hiddenErrors, read_only image2d_t hiddenStates,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha, float weightLambda)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	float reward = read_imagef(rewards, hiddenPosition).x;

	float state = read_imagef(hiddenStates, hiddenPosition).x;

	float error = read_imagef(hiddenErrors, hiddenPosition).x * state;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float visibleError = read_imagef(visibleErrors, visiblePosition).x;
				float visibleState = read_imagef(visibleStates, visiblePosition).x;

				float2 weight = (float2)(weightPrev.x + reward * weightPrev.y, weightPrev.y * weightLambda + weightAlpha * ((visibleError - weightPrev.x) * state));

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
			}
		}
}

// ----------------------------------------- Sparse Coder -----------------------------------------

void kernel scReconstructVisibleError(read_only image2d_t hiddenStates, read_only image2d_t visibleStates,
	write_only image2d_t reconstructionError, read_only image3d_t weights,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float recon = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

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

	write_imagef(reconstructionError, visiblePosition, (float4)(error));
}

void kernel scActivateFromReconstructionError(read_only image2d_t reconstructionError,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float sum = read_imagef(hiddenSummationTempBack, hiddenPosition).x;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float error = read_imagef(reconstructionError, visiblePosition).x;

				sum += weight * error;
			}
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum));
}

void kernel scSolveHidden(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenSpikesBack, write_only image2d_t hiddenSpikesFront, 
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront, 
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront, 
	read_only image2d_t hiddenThresholds, read_only image3d_t weightsLateral,
	int2 hiddenSize, int radius, float leak, float accum) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float excitation = read_imagef(hiddenSummationTemp, hiddenPosition).x;

	float statePrev = read_imagef(hiddenStatesBack, hiddenPosition).x;

	int2 fieldLowerBound = hiddenPosition - (int2)(radius);

	float inhibition = 0.0f;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			if (dx == 0 && dy == 0)
				continue;
			
			int2 otherPosition = hiddenPosition + (int2)(dx, dy);

			if (inBounds0(otherPosition, hiddenSize)) {
				int2 offset = otherPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weightsLateral, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float otherSpike = read_imagef(hiddenSpikesBack, otherPosition).x;

				inhibition += weight * otherSpike;
			}
		}

	float activation = read_imagef(hiddenActivationsBack, hiddenPosition).x;

	activation = (1.0f - leak) * activation + excitation - inhibition;

	float spike = 0.0f;

	float threshold = read_imagef(hiddenThresholds, hiddenPosition).x;

	if (activation > threshold) {
		spike = 1.0f;

		activation = 0.0f;
	}

	float state = (1.0f - accum) * statePrev + accum * spike;

	write_imagef(hiddenSpikesFront, hiddenPosition, (float4)(spike));
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(activation));
}

void kernel scLearnThresholds(read_only image2d_t hiddenThresholdsBack, write_only image2d_t hiddenThresholdsFront,
	read_only image2d_t hiddenStates,
	float thresholdAlpha, float activeRatio)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float thresholdPrev = read_imagef(hiddenThresholdsBack, hiddenPosition).x;

	float hiddenState = read_imagef(hiddenStates, hiddenPosition).x;

	float threshold = thresholdPrev + thresholdAlpha * (hiddenState - activeRatio);

	write_imagef(hiddenThresholdsFront, hiddenPosition, (float4)(threshold));
}

void kernel scLearnSparseCoderWeights(read_only image2d_t reconstructionError,
	read_only image2d_t hiddenStates, read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	float state = read_imagef(hiddenStates, hiddenPosition).x;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float error = read_imagef(reconstructionError, visiblePosition).x;

				float weight = weightPrev + weightAlpha * error * state;

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight));
			}
		}
}

void kernel scLearnSparseCoderWeightsTraces(read_only image2d_t reconstructionError,
	read_only image2d_t hiddenStates, read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	read_only image2d_t rewards,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha, float weightTraceLambda)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	float state = read_imagef(hiddenStates, hiddenPosition).x;

	float reward = read_imagef(rewards, hiddenPosition).x;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float error = read_imagef(reconstructionError, visiblePosition).x;

				float2 weight = (float2)(weightPrev.x + reward * weightPrev.y, weightPrev.y * weightTraceLambda + weightAlpha * state * error);

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f));
			}
		}
}

void kernel scLearnSparseCoderWeightsLateral(read_only image2d_t hiddenStates,
	read_only image3d_t weightsLateralBack, write_only image3d_t weightsLateralFront,
	int2 hiddenSize, int radius, float weightLateralAlpha, float activeRatioSquared)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 fieldLowerBound = hiddenPosition - (int2)(radius);

	float state = read_imagef(hiddenStates, hiddenPosition).x;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 otherPosition = hiddenPosition + (int2)(dx, dy);

			if (inBounds0(otherPosition, hiddenSize)) {
				int2 offset = otherPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weightPrev = read_imagef(weightsLateralBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float otherState = read_imagef(hiddenStates, otherPosition).x;

				float weight = fmax(0.0f, weightPrev + weightLateralAlpha * (state * otherState - activeRatioSquared));

				write_imagef(weightsLateralFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight));
			}
		}
}

// ----------------------------------------- Predictor -----------------------------------------

void kernel predErrorPropagate(read_only image2d_t targets, read_only image2d_t hiddenStatesPrev,
	write_only image2d_t errors, read_only image3d_t weights,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float error = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {	
					int2 offset = visiblePosition - fieldLowerBound;

					float predError = read_imagef(targets, hiddenPosition).x - read_imagef(hiddenStatesPrev, hiddenPosition).x;

					int wi = offset.y + offset.x * (radius * 2 + 1);

					float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
					error += predError * weight;
				}
			}
		}

	write_imagef(errors, visiblePosition, (float4)(error));
}

void kernel predActivate(read_only image2d_t visibleStates,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float sum = read_imagef(hiddenSummationTempBack, hiddenPosition).x;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float state = read_imagef(visibleStates, visiblePosition).x;

				sum += weight * state;
			}
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum));
}

void kernel predSolveHidden(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront,
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = read_imagef(hiddenSummationTemp, hiddenPosition).x;
	
	float s = sum;//sigmoid(sum);

	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(s));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(s));
}

void kernel predSolveHiddenThreshold(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront, 
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float activation = read_imagef(hiddenSummationTemp, hiddenPosition).x;

	float state = activation > 0.5f ? 1.0f : 0.0f;

	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(activation));
}

void kernel predLearnWeights(read_only image2d_t visibleStatesPrev, 
	read_only image2d_t targets, read_only image2d_t predictionsPrev, read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
	float target = read_imagef(targets, hiddenPosition).x;
	float predPrev = read_imagef(predictionsPrev, hiddenPosition).x;

	float alphaError = weightAlpha * (target - predPrev);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float state = read_imagef(visibleStatesPrev, visiblePosition).x;

				float weight = weightPrev + alphaError * state;

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight));
			}
		}
}

void kernel predLearnWeightsTraces(read_only image2d_t visibleStatesPrev, 
	read_only image2d_t targets, read_only image2d_t predictionsPrev, read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha, float weightLambda, float reward)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
	float target = read_imagef(targets, hiddenPosition).x;
	float predPrev = read_imagef(predictionsPrev, hiddenPosition).x;

	float error = target - predPrev;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float state = read_imagef(visibleStatesPrev, visiblePosition).x;

				float newTrace = weightPrev.y * weightLambda + weightAlpha * error * state;

				float2 weight = (float2)(weightPrev.x + reward * newTrace, newTrace);

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f));
			}
		}
}

// ----------------------------------------- Predictor Swarm -----------------------------------------

void kernel predErrorPropagateSwarm(read_only image2d_t targets, read_only image2d_t hiddenStatesPrev,
	write_only image2d_t errors, read_only image3d_t weights,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float error = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {	
					int2 offset = visiblePosition - fieldLowerBound;

					float predError = read_imagef(targets, hiddenPosition).x - read_imagef(hiddenStatesPrev, hiddenPosition).x;

					int wi = offset.y + offset.x * (radius * 2 + 1);

					float weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
					error += predError * weight;
				}
			}
		}

	write_imagef(errors, visiblePosition, (float4)(error));
}

void kernel predActivateSwarm(read_only image2d_t visibleStates,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float2 sum = read_imagef(hiddenSummationTempBack, hiddenPosition).xy;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xz;

				float state = read_imagef(visibleStates, visiblePosition).x;

				sum += weight * state;
			}
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum, 0.0f, 0.0f));
}

void kernel predSolveHiddenSwarm(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront,
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront,
	float noise, uint2 seed) 
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 123 + 5, get_global_id(1) * 24 + 2) * 45;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 sum = read_imagef(hiddenSummationTemp, hiddenPosition).xy;

	float s = sigmoid(sum.x);

	float2 state = (float2)(fmin(1.0f, fmax(0.0f, s + randNormal(&seedValue) * noise)), sum.y);
	
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state, 0.0f, 0.0f));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(s, sum.y, 0.0f, 0.0f));
}

void kernel predSolveHiddenThresholdSwarm(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront, 
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront,
	float noise, uint2 seed)  
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 45 + 25, get_global_id(1) * 56 + 24) * 6;

	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 sum = read_imagef(hiddenSummationTemp, hiddenPosition).xy;
	
	float s = sum.x > 0.5f ? 1.0f : 0.0f;

	float2 state = (float2)(randFloat(&seedValue) < noise ? 1.0f - s : s, sum.y);
	
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state, 0.0f, 0.0f));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(s, sum.y, 0.0f, 0.0f));
}

void kernel predLearnWeightsTracesSwarm(read_only image2d_t visibleStatesPrev, 
	read_only image2d_t targets, read_only image2d_t predictionStates, read_only image2d_t predictionActivationsPrev, read_only image2d_t predictionStatesPrev, read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float3 weightAlpha, float2 weightLambda, float reward, float gamma)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
	float target = read_imagef(targets, hiddenPosition).x;
	float2 state = read_imagef(predictionStates, hiddenPosition).xy;
	float predActPrev = read_imagef(predictionActivationsPrev, hiddenPosition).x;
	float2 predPrev = read_imagef(predictionStatesPrev, hiddenPosition).xy;

	float predError = target - predActPrev;

	float randError = predPrev.x - predActPrev;

	float tdError = reward + gamma * state.y - predPrev.y;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float4 weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0));

				float statePrev = read_imagef(visibleStatesPrev, visiblePosition).x;

				//float oneMinusStatePrev = 1.0f - statePrev;

				float newYTrace = weightPrev.y * weightLambda.x + weightAlpha.x * randError * statePrev;
				float newWTrace = weightPrev.w * weightLambda.y + weightAlpha.y * statePrev;

				float4 weight = (float4)(weightPrev.x + weightAlpha.z * predError * statePrev + (tdError > 0.0f ? 1.0f : 0.0f) * newYTrace, newYTrace,
						weightPrev.z + tdError * newWTrace, newWTrace);

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), weight);
			}
		}
}

// ----------------------------------------- Predictor Swarm -----------------------------------------

void kernel swarmQPropagateToHiddenError(read_only image3d_t weights, write_only image2d_t hiddenErrors,
	int2 qSize, int2 hiddenSize, float2 qToHidden, float2 hiddenToQ, int radius, int2 reverseQRadii)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 qPositionCenter = (int2)(hiddenPosition.x * hiddenToQ.x + 0.5f, hiddenPosition.y * hiddenToQ.y + 0.5f);
	
	float2 error = (float2)(0.0f);

	for (int dx = -reverseQRadii.x; dx <= reverseQRadii.x; dx++)
		for (int dy = -reverseQRadii.y; dy <= reverseQRadii.y; dy++) {
			int2 qPosition = qPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(qPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(qPosition.x * qToHidden.x + 0.5f, qPosition.y * qToHidden.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(hiddenPosition, fieldLowerBound, fieldUpperBound)) {	
					int2 offset = hiddenPosition - fieldLowerBound;

					//float qState = read_imagef(qStates, qPosition).x;

					int wi = offset.y + offset.x * (radius * 2 + 1);

					float2 weight = read_imagef(weights, (int4)(qPosition.x, qPosition.y, wi, 0)).xz;
				
					error += weight;
				}
			}
		}

	write_imagef(hiddenErrors, hiddenPosition, (float4)(error.x, error.y, 0.0f, 0.0f));
}

void kernel swarmQPropagateToHiddenTD(read_only image2d_t qStates, read_only image2d_t qStatesPrev, 
	write_only image2d_t hiddenTDErrors,
	int2 qSize, int2 hiddenSize, float2 qToHidden, float2 hiddenToQ, int radius, int2 reverseQRadii,
	float reward, float gamma)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 qPositionCenter = (int2)(hiddenPosition.x * hiddenToQ.x + 0.5f, hiddenPosition.y * hiddenToQ.y + 0.5f);
	
	float sum = 0.0f;
	float div = 0.0f;

	for (int dx = -reverseQRadii.x; dx <= reverseQRadii.x; dx++)
		for (int dy = -reverseQRadii.y; dy <= reverseQRadii.y; dy++) {
			int2 qPosition = qPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(qPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(qPosition.x * qToHidden.x + 0.5f, qPosition.y * qToHidden.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(hiddenPosition, fieldLowerBound, fieldUpperBound)) {	
					float qState = read_imagef(qStates, qPosition).x;
					float qStatePrev = read_imagef(qStatesPrev, qPosition).x;

					float tdError = reward + gamma * qState - qStatePrev;

					sum += tdError;
					div += 1.0f;
				}
			}
		}

	write_imagef(hiddenTDErrors, hiddenPosition, (float4)(sum / fmax(1.0f, div)));
}

void kernel swarmPredictAction(read_only image2d_t hiddenStatesFeedForward, read_only image2d_t actionsFeedBack,
	read_only image3d_t weights, write_only image2d_t predictedAction,
	int2 hiddenSize, float2 visibleToHidden, int radius)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float sum = 0.0f;

	int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				int2 offset = hiddenPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weight = read_imagef(weights, (int4)(visiblePosition.x, visiblePosition.y, wi, 0)).xy;

				float hsff = read_imagef(hiddenStatesFeedForward, hiddenPosition).x;
				float afb = read_imagef(actionsFeedBack, hiddenPosition).x;

				sum += weight.x * hsff + weight.y * afb;
			}
		}

	write_imagef(predictedAction, visiblePosition, (float4)(tanh(sum)));
}

void kernel swarmInitSummation(read_only image2d_t hiddenBiases, write_only image2d_t hiddenSummationTempFront) {
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 biases = read_imagef(hiddenBiases, hiddenPosition).xz;
	
	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(biases.x, biases.y, 0.0f, 0.0f));
}

void kernel swarmQActivateToHidden(read_only image2d_t visibleStates,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float2 sum = read_imagef(hiddenSummationTempBack, hiddenPosition).xy;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xz;

				float state = read_imagef(visibleStates, visiblePosition).x;

				sum += weight * state;
			}
		}

	write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum.x, sum.y, 0.0f, 0.0f));
}

void kernel swarmQSolveHidden(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesFeedForward, read_only image2d_t actionsFeedBack,
	write_only image2d_t hiddenStates) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 sum = read_imagef(hiddenSummationTemp, hiddenPosition).xy;

	float hsff = read_imagef(hiddenStatesFeedForward, hiddenPosition).x;
	float afb = read_imagef(actionsFeedBack, hiddenPosition).x;

	write_imagef(hiddenStates, hiddenPosition, (float4)(tanh(sum.x) * hsff, tanh(sum.y) * afb, 0.0f, 0.0f));
}

void kernel swarmHiddenPropagateToVisibleAction(read_only image2d_t hiddenErrors, read_only image2d_t hiddenStates,
	read_only image3d_t weights, read_only image2d_t actionsBack, write_only image2d_t actionsFront,
	int2 hiddenSize, int2 visibleSize, float2 hiddenToVisible, float2 visibleToHidden, int radius, int2 reverseRadii,
	float actionAlpha)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float error = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {	
					int2 offset = visiblePosition - fieldLowerBound;

					float2 hiddenState = read_imagef(hiddenStates, hiddenPosition).xy;
					float2 hiddenError = read_imagef(hiddenErrors, hiddenPosition).xy;

					int wi = offset.y + offset.x * (radius * 2 + 1);

					float2 weight = read_imagef(weights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xz;
				
					error += dot((1.0f - hiddenState * hiddenState) * hiddenError, weight);
				}
			}
		}

	float prevAction = read_imagef(actionsBack, visiblePosition).x;

	float nextAction = fmin(1.0f, fmax(-1.0f, prevAction + actionAlpha * (error > 0.0f ? 1.0f : -1.0f)));

	write_imagef(actionsFront, visiblePosition, (float4)(nextAction));
}

void kernel swarmExploration(read_only image2d_t actions,
	write_only image2d_t actionsExploratory, float expPert, float expBreak, uint2 seed)  
{
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 45 + 25, get_global_id(1) * 56 + 24) * 6;

	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float action = read_imagef(actions, position).x;
	
	write_imagef(actionsExploratory, position, (float4)(randFloat(&seedValue) < expBreak ? randFloat(&seedValue) * 2.0f - 1.0f : fmin(1.0f, fmax(0.0f, action + expPert * randNormal(&seedValue)))));
}

void kernel swarmQActivateToQ(read_only image2d_t hiddenStates,
	read_only image3d_t weights, write_only image2d_t qStates,
	int2 hiddenSize, float2 qToHidden, int radius)
{
	int2 qPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(qPosition.x * qToHidden.x + 0.5f, qPosition.y * qToHidden.y + 0.5f);
	
	float sum = 0.0f;

	int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				int2 offset = hiddenPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weight = read_imagef(weights, (int4)(qPosition.x, qPosition.y, wi, 0)).xy;

				float2 state = read_imagef(hiddenStates, hiddenPosition).xy;

				sum += dot(weight, state);
			}
		}

	write_imagef(qStates, qPosition, (float4)(sum));
}

void kernel swarmQLearnVisibleWeightsTraces(read_only image2d_t actionsExploratory, 
	read_only image2d_t hiddenErrors, read_only image2d_t hiddenTD, read_only image2d_t hiddenStates,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float alpha, float lambda)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
	float tdError = read_imagef(hiddenTD, hiddenPosition).x;

	float2 hiddenState = read_imagef(hiddenStates, hiddenPosition).xy;

	float2 hiddenError = read_imagef(hiddenErrors, hiddenPosition).xy;

	float2 error = hiddenError * (1.0f - hiddenState * hiddenState);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float4 weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0));

				float state = read_imagef(actionsExploratory, visiblePosition).x;

				float4 weight = (float4)(weightPrev.x + tdError * weightPrev.y, lambda * weightPrev.y + alpha * error.x * state,
						weightPrev.z + tdError * weightPrev.w, lambda * weightPrev.w + alpha * error.y * state);

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), weight);
			}
		}
}

void kernel swarmStartLearnWeights(read_only image2d_t actions, read_only image2d_t predictedAction,
	read_only image2d_t hiddenStatesFeedForward, read_only image2d_t actionsFeedBack,
	read_only image3d_t weightsPrev, write_only image3d_t weights,
	int2 hiddenSize, float2 visibleToHidden, int radius,
	float alpha)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float alphaError = alpha * (read_imagef(actions, visiblePosition).x - read_imagef(predictedAction, visiblePosition).x);

	int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				int2 offset = hiddenPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weightPrev = read_imagef(weightsPrev, (int4)(visiblePosition.x, visiblePosition.y, wi, 0)).xy;

				float hsff = read_imagef(hiddenStatesFeedForward, hiddenPosition).x;
				float afb = read_imagef(actionsFeedBack, hiddenPosition).x;

				float2 weight = weightPrev + alphaError * (float2)(hsff, afb);
				
				write_imagef(weights, (int4)(visiblePosition.x, visiblePosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f));
			}
		}
}

void kernel swarmQLearnHiddenWeightsTraces(read_only image2d_t hiddenStates,
	read_only image2d_t qStates, read_only image2d_t qStatesPrev,
	read_only image3d_t weightsPrev, write_only image3d_t weights,
	int2 hiddenSize, float2 qToHidden, int radius,
	float alpha, float lambda, float reward, float gamma)
{
	int2 qPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(qPosition.x * qToHidden.x + 0.5f, qPosition.y * qToHidden.y + 0.5f);
	
	float qState = read_imagef(qStates, qPosition).x;
	float qStatePrev = read_imagef(qStatesPrev, qPosition).x;

	float tdError = reward + gamma * qState - qStatePrev;

	int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				int2 offset = hiddenPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float4 weightPrev = read_imagef(weightsPrev, (int4)(qPosition.x, qPosition.y, wi, 0));

				float2 state = read_imagef(hiddenStates, hiddenPosition).xy;

				float4 weight = (float4)(weightPrev.x + tdError * weightPrev.y, weightPrev.y * lambda + alpha * state.x,
					weightPrev.z + tdError * weightPrev.w, weightPrev.w * lambda + alpha * state.y);
				
				write_imagef(weights, (int4)(qPosition.x, qPosition.y, wi, 0), weight);
			}
		}
}

void kernel swarmQLearnHiddenBiasesTraces(read_only image2d_t hiddenTD, read_only image2d_t hiddenErrors,
	read_only image2d_t biasesBack, write_only image2d_t biasesFront,
	float alpha, float lambda)  
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float4 biasPrev = read_imagef(biasesBack, hiddenPosition);

	float tdError = read_imagef(hiddenTD, hiddenPosition).x;

	float2 error = read_imagef(hiddenErrors, hiddenPosition).xy;

	float4 bias = (float4)(biasPrev.x + tdError * biasPrev.y, biasPrev.y * lambda + alpha * error.x,
		biasPrev.z + tdError * biasPrev.w, biasPrev.w * lambda + alpha * error.y);
				
	write_imagef(biasesFront, hiddenPosition, bias);
}

// ----------------------------------------- Predictive Hierarchy -----------------------------------------

void kernel phBaseLineUpdate(read_only image2d_t errorsCurrent, read_only image2d_t hiddenStates,
	read_only image2d_t baseLinesBack, write_only image2d_t baseLinesFront, write_only image2d_t rewards,
	float decay, float sensitivity)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float state = read_imagef(hiddenStates, position).x;

	float error = read_imagef(errorsCurrent, position).x;

	float correctness = 1.0f - fabs(error);

	float baseLinePrev = read_imagef(baseLinesBack, position).x;

	float reward = (baseLinePrev - correctness) > 0.0f ? 1.0f : 0.0f;

	float baseLine = (1.0f - decay) * baseLinePrev + decay * correctness;

	write_imagef(baseLinesFront, position, (float4)(baseLine));
	write_imagef(rewards, position, (float4)(reward));
}

void kernel phBaseLineUpdateSumError(read_only image2d_t errorsLower, read_only image2d_t errorsCurrent, read_only image2d_t hiddenStates,
	read_only image2d_t baseLinesBack, write_only image2d_t baseLinesFront, write_only image2d_t rewards,
	float decay, float sensitivity)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float state = read_imagef(hiddenStates, position).x;

	float error = read_imagef(errorsLower, position).x + read_imagef(errorsCurrent, position).x;

	float correctness = 1.0f - fabs(error);

	float baseLinePrev = read_imagef(baseLinesBack, position).x;

	float reward = (baseLinePrev - correctness) > 0.0f ? 1.0f : 0.0f;

	float baseLine = (1.0f - decay) * baseLinePrev + decay * correctness;

	write_imagef(baseLinesFront, position, (float4)(baseLine));
	write_imagef(rewards, position, (float4)(reward));
}

void kernel phInhibit(read_only image2d_t activations,
	write_only image2d_t states,
	int2 size, int radius, float activeRatio)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float activation = read_imagef(activations, position).x;

	int2 fieldLowerBound = position - (int2)(radius);

	float inhibition = 0.0f;

	float counter = 0.0f;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			if (dx == 0 && dy == 0)
				continue;
			
			int2 otherPosition = position + (int2)(dx, dy);

			if (inBounds0(otherPosition, size)) {
				float otherActivation = read_imagef(activations, otherPosition).x;

				inhibition += otherActivation >= activation ? 1.0f : 0.0f;

				counter++;
			}
		}

	float state = inhibition < (counter * activeRatio) ? 1.0f : 0.0f;

	write_imagef(states, position, (float4)(state));
}

void kernel phModulate(read_only image2d_t inputsLeft, read_only image2d_t inputsRight,
	write_only image2d_t states, float minAttention)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float left = read_imagef(inputsLeft, position).x;
	float right = read_imagef(inputsRight, position).x;

	write_imagef(states, position, (float4)(left * (minAttention + (1.0f - minAttention) * (right * 0.5f + 0.5f))));
}

void kernel phCopyAction(read_only image2d_t source, write_only image2d_t destination) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float s = read_imagef(source, position).x;
	
	write_imagef(destination, position, (float4)(s));
}

// ----------------------------------------- Q Route -----------------------------------------

void kernel qForward(read_only image2d_t hiddenStates, read_only image3d_t qWeights, read_only image2d_t qBiases, read_only image2d_t qStatesPrev, write_only image2d_t qStatesFront, write_only image2d_t qActivationsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float eluAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float sum = 0.0f;//read_imagef(qBiases, hiddenPosition).x;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(qWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float state = read_imagef(qStatesPrev, visiblePosition).x;

				sum += weight * state;
			}
		}

	float hiddenState = read_imagef(hiddenStates, hiddenPosition).x;

	float s = elu(sum, eluAlpha);
	float state = s * hiddenState;//elu(sum, eluAlpha) * hiddenState;
	
	write_imagef(qStatesFront, hiddenPosition, (float4)(state));
	write_imagef(qActivationsFront, hiddenPosition, (float4)(s));
}

void kernel qForwardFirstLayer(read_only image2d_t hiddenStates, read_only image2d_t originalActions, read_only image3d_t qWeights, read_only image2d_t qBiases, read_only image2d_t qStatesPrev, write_only image2d_t qStatesFront, write_only image2d_t qActivationsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float eluAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float sum = 0.0f;//read_imagef(qBiases, hiddenPosition).x;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weight = read_imagef(qWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xz;

				float state = read_imagef(qStatesPrev, visiblePosition).x;

				float original = read_imagef(originalActions, visiblePosition).x;

				sum += weight.x * state + weight.y * (1.0f - state);
			}
		}

	float hiddenState = read_imagef(hiddenStates, hiddenPosition).x;

	float s = elu(sum, eluAlpha);
	float state = s * hiddenState;//elu(sum, eluAlpha) * hiddenState;
	
	write_imagef(qStatesFront, hiddenPosition, (float4)(state));
	write_imagef(qActivationsFront, hiddenPosition, (float4)(s));
}

void kernel qBackward(read_only image2d_t hiddenStates, read_only image3d_t qWeights, read_only image2d_t qStates, read_only image2d_t qActivations, read_only image2d_t qErrorsNext, write_only image2d_t qErrors,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii,
	float eluAlpha)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float sum = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {	
					int2 offset = visiblePosition - fieldLowerBound;

					float errorNext = read_imagef(qErrorsNext, hiddenPosition).x;

					int wi = offset.y + offset.x * (radius * 2 + 1);

					float weight = read_imagef(qWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
					sum += errorNext * weight;
				}
			}
		}

	float hiddenState = read_imagef(hiddenStates, visiblePosition).x;

	float state = read_imagef(qStates, visiblePosition).x;

	float activation = read_imagef(qActivations, visiblePosition).x;

	float error = sum * elud(activation, eluAlpha) * hiddenState;

	write_imagef(qErrors, visiblePosition, (float4)(error));
}

void kernel qBackwardFirstLayer(read_only image3d_t qWeights, read_only image2d_t qErrorsNext, write_only image2d_t qErrors,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
	int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);
	
	float sum = 0.0f;

	for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
		for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);
		
			if (inBounds0(hiddenPosition, hiddenSize)) {
				// Next layer node's receptive field
				int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

				int2 fieldLowerBound = fieldCenter - (int2)(radius);
				int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds
		
				// Check for containment
				if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {	
					int2 offset = visiblePosition - fieldLowerBound;

					float errorNext = read_imagef(qErrorsNext, hiddenPosition).x;

					int wi = offset.y + offset.x * (radius * 2 + 1);

					float2 weight = read_imagef(qWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xz;
				
					sum += errorNext * weight.x - errorNext * weight.y;
				}
			}
		}

	write_imagef(qErrors, visiblePosition, (float4)(sum));
}

void kernel qWeightUpdate(read_only image2d_t qStatesPrev, read_only image2d_t qStates, read_only image2d_t qErrors,
	read_only image3d_t qWeightsBack, write_only image3d_t qWeightsFront,
	read_only image2d_t qBiasesBack, write_only image2d_t qBiasesFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float alpha, float biasAlpha, float gammaLambda, float tdError)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float state = read_imagef(qStates, hiddenPosition).x;

	float error = read_imagef(qErrors, hiddenPosition).x;
	
	// Bias
	float2 biasPrev = read_imagef(qBiasesBack, hiddenPosition).xy;

	float2 bias = (float2)(biasPrev.x + alpha * tdError * biasPrev.y, biasPrev.y * gammaLambda + error);
	//float2 bias = (float2)(biasPrev.x + biasAlpha * (0.5f - state), biasPrev.y * gammaLambda + error);

	write_imagef(qBiasesFront, hiddenPosition, (float4)(bias, 0.0f, 0.0f));

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weightPrev = read_imagef(qWeightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float statePrev = read_imagef(qStatesPrev, visiblePosition).x;

				//float oneMinusStatePrev = 1.0f - statePrev;

				float2 weight = (float2)(weightPrev.x + alpha * tdError * weightPrev.y, weightPrev.y * gammaLambda + error * statePrev);

				write_imagef(qWeightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f));
			}
		}
}

void kernel qWeightUpdateFirstLayer(read_only image2d_t qStatesPrev, read_only image2d_t originalActions, read_only image2d_t qStates, read_only image2d_t qErrors,
	read_only image3d_t qWeightsBack, write_only image3d_t qWeightsFront,
	read_only image2d_t qBiasesBack, write_only image2d_t qBiasesFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float alpha, float biasAlpha, float gammaLambda, float tdError)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float state = read_imagef(qStates, hiddenPosition).x;

	float error = read_imagef(qErrors, hiddenPosition).x;
	
	// Bias
	float2 biasPrev = read_imagef(qBiasesBack, hiddenPosition).xy;

	float2 bias = (float2)(biasPrev.x + alpha * tdError * biasPrev.y, biasPrev.y * gammaLambda + error);
	//float2 bias = (float2)(biasPrev.x + biasAlpha * (0.5f - state), biasPrev.y * gammaLambda + error);

	write_imagef(qBiasesFront, hiddenPosition, (float4)(bias, 0.0f, 0.0f));

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float4 weightPrev = read_imagef(qWeightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0));

				float statePrev = read_imagef(qStatesPrev, visiblePosition).x;
				float original = read_imagef(originalActions, visiblePosition).x;
				//float oneMinusStatePrev = 1.0f - statePrev;

				float4 weight = (float4)(weightPrev.x + alpha * tdError * weightPrev.y, weightPrev.y * gammaLambda + error * statePrev,
					weightPrev.z + alpha * tdError * weightPrev.w, weightPrev.w * gammaLambda + error * (1.0f - statePrev));

				write_imagef(qWeightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), weight);
			}
		}
}
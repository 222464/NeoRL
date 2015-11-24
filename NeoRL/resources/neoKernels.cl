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
	if (x < 1.0f)
		return x > 0.0f ? x : x * leak;

	return (x - 1.0f) * leak + 1.0f;
}

float relud(float x, float leak) {
	return x > 0.0f && x < 1.0f ? 1.0f : leak;
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

// Initialize a random uniform 2D image (XZ fields)
void kernel randomUniform2DXZ(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, position, (float4)(v.x, 0.0f, v.y, 0.0f));
}

// Initialize a random uniform 3D image (XZ fields)
void kernel randomUniform3DXZ(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, (int4)(position, 0), (float4)(v.x, 0.0f, v.y, 0.0f));
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

				float weight = weightPrev + weightAlpha * state * error * state;

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
	
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(sum));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(sum));
}

void kernel predSolveHiddenThreshold(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront, 
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float sum = read_imagef(hiddenSummationTemp, hiddenPosition).x;

	float state = fmin(1.0f, fmax(0.0f, sum));
	
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(sum));
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
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 sum = read_imagef(hiddenSummationTemp, hiddenPosition).xy;

	float2 state = (float2)(sigmoid(sum.x) * 2.0f - 1.0f, sum.y);
	
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state, 0.0f, 0.0f));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(state, 0.0f, 0.0f));
}

void kernel predSolveHiddenThresholdSwarm(read_only image2d_t hiddenSummationTemp,
	read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront, 
	read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront) 
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float2 sum = read_imagef(hiddenSummationTemp, hiddenPosition).xy;
	
	float2 state = (float2)(fmin(1.0f, fmax(0.0f, sum.x)), sum.y);
	
	write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state, 0.0f, 0.0f));
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(sum, 0.0f, 0.0f));
}

void kernel predLearnWeightsTracesSwarm(read_only image2d_t visibleStatesPrev, 
	read_only image2d_t targets, read_only image2d_t predictions, read_only image2d_t predictionsPrev, read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float2 weightAlpha, float2 weightLambda, float reward, float gamma)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
	float target = read_imagef(targets, hiddenPosition).x;
	float2 pred = read_imagef(predictions, hiddenPosition).xy;
	float2 predPrev = read_imagef(predictionsPrev, hiddenPosition).xy;

	float error = target - predPrev.x;

	float tdError = reward + gamma * pred.y - predPrev.y;

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float4 weightPrev = read_imagef(weightsBack, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0));

				float statePrev = read_imagef(visibleStatesPrev, visiblePosition).x;

				float newYTrace = weightPrev.y * weightLambda.x + weightAlpha.x * error * statePrev;
				float newWTrace = weightPrev.w * weightLambda.y + weightAlpha.y * statePrev;

				float4 weight = (float4)(weightPrev.x + fmax(0.0f, tdError) * newYTrace, newYTrace,
						weightPrev.z + tdError * newWTrace, newWTrace);

				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), weight);
			}
		}
}

// ----------------------------------------- Predictive Hierarchy -----------------------------------------

void kernel phBaseLineUpdate(read_only image2d_t targets, read_only image2d_t predictionsPrev,
	read_only image2d_t baseLinesBack, write_only image2d_t baseLinesFront, write_only image2d_t rewards,
	float decay, float sensitivity)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	float target = read_imagef(targets, position).x;

	float pred = read_imagef(predictionsPrev, position).x;

	float error = target - pred;

	float error2 = error * error;

	float baseLinePrev = read_imagef(baseLinesBack, position).x;

	float reward = sigmoid(sensitivity * (error2 - baseLinePrev));

	float baseLine = (1.0f - decay) * baseLinePrev + decay * error2;

	write_imagef(baseLinesFront, position, (float4)(baseLine));
	write_imagef(rewards, position, (float4)(reward));
}

// ----------------------------------------- Q Route -----------------------------------------

void kernel qForward(read_only image2d_t hiddenStates, read_only image3d_t qWeights, read_only image2d_t qBiases, read_only image2d_t qStatesPrev, write_only image2d_t qStatesFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float reluLeak)
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

	float state = relu(sum, reluLeak) * hiddenState;
	
	write_imagef(qStatesFront, hiddenPosition, (float4)(state));
}

void kernel qBackward(read_only image2d_t hiddenStates, read_only image3d_t qWeights, read_only image2d_t qStates, read_only image2d_t qErrorsNext, write_only image2d_t qErrors,
	int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii,
	float reluLeak)
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

	float error = sum * relud(state, reluLeak) * hiddenState;

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

					float weight = read_imagef(qWeights, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;
				
					sum += errorNext * weight;
				}
			}
		}

	write_imagef(qErrors, visiblePosition, (float4)(sum));
}

void kernel qWeightUpdate(read_only image2d_t qStatesPrev, read_only image2d_t qStates, read_only image2d_t qErrors,
	read_only image3d_t qWeightsBack, write_only image3d_t qWeightsFront,
	read_only image2d_t qBiasesBack, write_only image2d_t qBiasesFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float alpha, float gammaLambda, float tdError)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);
	
	float state = read_imagef(qStates, hiddenPosition).x;

	float error = read_imagef(qErrors, hiddenPosition).x;
	
	// Bias
	float2 biasPrev = read_imagef(qBiasesBack, hiddenPosition).xy;

	float2 bias = (float2)(biasPrev.x + alpha * (tdError * biasPrev.y + (0.5f - state)), biasPrev.y * gammaLambda + error);

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

				float2 weight = (float2)(weightPrev.x + alpha * (tdError * weightPrev.y + (0.5f - state) * statePrev), weightPrev.y * gammaLambda + error * statePrev);

				write_imagef(qWeightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f));
			}
		}
}
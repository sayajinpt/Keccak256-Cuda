#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>

#define BLOCK_SIZE 256

__global__ void generateRandomHexString(char* output, curandState* state, unsigned long long seed, int numStrings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
    char hexChars[] = "0123456789ABCDEF";

    for (int j = 0; j < numStrings; ++j) {
        int stringStartIdx = j * 65;
        for (int i = 0; i < 64; ++i) {
            output[stringStartIdx + idx * 64 + i] = hexChars[curand(&state[idx]) % 16];
        }
        output[stringStartIdx + idx * 64 + 64] = '\0';
    }
}

extern "C" void cudaGenerateRandomHexStrings(char* output, int numStrings) {
    char* d_output;
    cudaMalloc((void**)&d_output, sizeof(char) * numStrings * 65);
    curandState* d_state;
    cudaMalloc((void**)&d_state, sizeof(curandState) * BLOCK_SIZE);

    unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

    generateRandomHexString<<<(numStrings + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_output, d_state, seed, numStrings);

    cudaMemcpy(output, d_output, sizeof(char) * numStrings * 65, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_state);
}

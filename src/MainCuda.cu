#pragma once
#include <iostream>
#include <math.h>
#include "glm/glm.hpp"
#include "headers/MainCuda.h"

__global__
void add(int n, float *x, float *y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i+=stride)
        y[i] = x[i] + y[i];
}


void MainCuda::doCalculation() {
    glm::vec3 a;
    a.x = 3;
    std::cout << "A is: " << a.x << std::endl;

    int N = 1<<20; // 1M elements

    // Allocate Unified Memory -- accessible from CPU or GPU
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 256>>>(N, x, y);
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    std::cout << y[0] << std::endl;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
}
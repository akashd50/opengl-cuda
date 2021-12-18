#pragma once
#include <iostream>
#include <math.h>
#include "glm/glm.hpp"
#include "headers/MainCuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

__global__
void add(int n, float *x, float *y)
{

    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    int index = threadIdx.x;
    printf("Index %d", index);
    int stride = blockDim.x;
    for (int i = index; i < n; i+=stride)
        y[i] = x[i] + y[i];
}

__global__
void textureCompute(cudaSurfaceObject_t image)
{

    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    unsigned int x = threadIdx.x;
    unsigned int y = blockIdx.x;
    //printf("Index (%d %d)", x, y);
    uchar4 color = make_uchar4(x / 2, y / 2, 0, 127);
    surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
}

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void MainCuda::texImageTest(Texture* texture) {
    struct cudaGraphicsResource *vbo_res;
    // register this texture with CUDA
    //cudaGraphicsGLRegisterImage(&vbo_res, texture->getTextureId(),GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
    check(cudaGraphicsGLRegisterImage(&vbo_res, texture->getTextureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    check(cudaGraphicsMapResources(1, &vbo_res));

    cudaArray_t viewCudaArray;
    check(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, vbo_res, 0, 0));

    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

    cudaSurfaceObject_t viewCudaSurfaceObject;
    check(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

    textureCompute<<<512, 512>>>(viewCudaSurfaceObject);

//    cudaArray *array;
//    cudaGraphicsMapResources(1, &vbo_res, 0);
//    cudaGraphicsSubResourceGetMappedArray(&array, vbo_res, 0,0);
//
//    texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texRef;
//    cudaBindTextureToArray(texRef, (cudaArray *)array));
//    texRef.filterMode = cudaFilterModeLinear;
}

void MainCuda::doCalculation() {

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Thread Blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Max Threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }

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
    add<<<1, 1024>>>(N, x, y);
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
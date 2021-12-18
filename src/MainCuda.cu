#pragma once
#include <iostream>
#include <math.h>
#include "glm/glm.hpp"
#include "headers/MainCuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>
#include <vector_types.h>
#include <math_functions.h>

__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

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

__device__ float3 cast_ray(unsigned int x, unsigned int y, int width, int height) {
    float d = 1.0;
    float fov = 60.0;
    float aspect_ratio = ((float)width) / ((float)height);
    float h = d * (float)tan((3.1415 * fov) / 180.0 / 2.0);
    float w = h * aspect_ratio;

    float top = h;
    float bottom = -h;
    float left = -w;
    float right = w;

    float u = left + (right - left) * float(x) / ((float)width);
    float v = bottom + (top - bottom) * (((float)height) - float(y)) / ((float)height);
    return make_float3(u, v, -d);
}

__device__ const float MIN_T = -9999.0;
__device__ const float HIT_T_OFFSET = 0.01;
//
__device__ float check_hit_on_sphere(float3 eye, float3 ray, float3 center, float radius) {
    float3 center_2_eye = eye - center;
    float ray_dot_ray = dot(ray, ray);
    float discriminant = pow(dot(ray, center_2_eye), 2) - ray_dot_ray * (dot(center_2_eye, center_2_eye) - pow(radius, 2));

    if (discriminant > 0) {
        discriminant = sqrt(discriminant);
        float init = -dot(ray, center_2_eye);
        float t1 = (init + discriminant) / ray_dot_ray;
        float t2 = (init - discriminant) / ray_dot_ray;

        float mint = min(t1, t2);
        if (mint < HIT_T_OFFSET) {
            return max(t1, t2);
        }
        return mint;
    }
    else if (discriminant == 0) {
        float init = -dot(ray, center_2_eye);
        float t1 = init / ray_dot_ray;
        return t1;
    }
    return MIN_T;
}

__global__
void textureCompute(cudaSurfaceObject_t image)
{
    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    unsigned int x = threadIdx.x;
    unsigned int y = blockIdx.x;

    float3 sphere = make_float3(0.0, 0.0, -5.0);
    float3 eye = make_float3(0.0, 0.0, 0.0);
    float radius = 0.5;
    float3 ray = cast_ray(x, y, 512, 512) - eye;
    float sphereHit = check_hit_on_sphere(eye, ray, sphere, radius);
    uchar4 color;
    if (sphereHit >= 0 && sphereHit != MIN_T) {
        color = make_uchar4(255, 0, 0, 255);
    } else {
        color = make_uchar4(0, 0, 0, 255);
    }

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
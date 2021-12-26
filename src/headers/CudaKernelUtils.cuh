#pragma once
#include "Texture.h"
#include <vector_types.h>
#include <vector>
#include <surface_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include "cuda_allocation_utils.h"
#include <curand_kernel.h>

//----------------------------------------------------------------------------------------------------------------------

__device__ const float MIN_T = -999999.0;
__device__ const float MAX_T = 999999.0;
__device__ const float HIT_T_OFFSET = 0.0001;
__device__ const float HIT_T_OFFSET_1 = 0.00005;

class MinMaxT {
public:
    float minT, maxT;
    __device__ MinMaxT(float _minT, float _maxT): minT(_minT), maxT(_maxT) {}
};

//----------------------------------------------------------------------------------------------------------------------

class HitInfo {
public:
    CudaRTObject* object;
    float t;
    float3 hitPoint, hitNormal, color;
    int index;
    __device__ HitInfo(): t(MAX_T) {}
    __device__ HitInfo(CudaRTObject* _object, float _t) : object(_object), t(_t) {}
    __device__ bool isHit() {
        return t != MAX_T;
    }
};

//----------------------------------------------------------------------------------------------------------------------

template <class T>
class Stack {
public:
    T* stack;
    int pointer;
    __device__ bool init() {
        stack = (T*)malloc(15 * sizeof(T));
        pointer = 0;
        return stack != NULL;
    }

    __device__ void push(T val) {
        if (pointer >= 15) {
            printf("Stack full\n");
        }
        stack[pointer++] = val;
    }

    __device__ void pop() {
        pointer--;
    }

    __device__ T top() {
        return stack[pointer - 1];
    }

    __device__ bool empty() {
        return pointer == 0;
    }

    __device__ int size() {
        return pointer;
    }

    __device__ void clean() {
        free(stack);
        pointer = 0;
    }
};

//----------------------------------------------------------------------------------------------------------------------

struct CudaThreadData {
    curandState randState;
    int randIndex;
    bool debug;
    CudaScene* scene;
};

//----------------------------------------------------------------------------------------------------------------------

class CudaKernelUtils {
private:
    cudaSurfaceObject_t viewCudaSurfaceObject;
public:
    CudaKernelUtils();
    ~CudaKernelUtils();
    void initializeRenderSurface(Texture* texture);
    void CudaKernelUtils::renderScene(CudaScene* cudaScene, int blockSize, int numThreads, int startRowIndex, int startColIndex);
    void onClick(int x, int y, CudaScene* cudaScene);
    void deviceInformation();
};

//----------------------------------------------------------------------------------------------------------------------

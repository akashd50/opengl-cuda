#pragma once
#include "Texture.h"
#include <vector_types.h>
#include <vector>
#include <surface_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include "cuda_allocation_utils.h"

//----------------------------------------------------------------------------------------------------------------------

__device__ const float MIN_T = -9999.0;
__device__ const float MAX_T = 999999.0;
__device__ const float HIT_T_OFFSET = 0.01;
__device__ const float HIT_T_OFFSET_1 = 0.0001;

class HitInfo {
public:
    CudaRTObject* object;
    float t;
    float3 hitPoint, hitNormal;
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
    __device__ void init() {
        stack = (T*)malloc(15 * sizeof(T));
        pointer = 0;
    }

    __device__ void push(T val) {
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

class CudaKernelUtils {
private:
    cudaSurfaceObject_t viewCudaSurfaceObject;
public:
    CudaKernelUtils();
    ~CudaKernelUtils();
    void initializeRenderSurface(Texture* texture);
    void renderScene(CudaScene* cudaScene);
    void onClick(int x, int y, CudaScene* cudaScene);
    void deviceInformation();
};

//----------------------------------------------------------------------------------------------------------------------

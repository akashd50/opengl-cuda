#pragma once
#include "Texture.h"
#include <vector_types.h>
#include <vector>
#include <surface_functions.h>
#include <surface_indirect_functions.h>
#include <cuda_runtime.h>
#include "cuda_helper_utils.h"

//----------------------------------------------------------------------------------------------------------------------

__device__ const float MIN_T = -9999.0;
__device__ const float MAX_T = 999999.0;
__device__ const float HIT_T_OFFSET = 0.01;

class HitInfo {
public:
    CudaRTObject* object;
    float t;
    float3 hitPoint;
    int index;
    __device__ HitInfo(): t(MAX_T) {}
    __device__ HitInfo(CudaRTObject* _object, float _t) : object(_object), t(_t) {}
    __device__ bool isHit() {
        return t != MAX_T;
    }
};

//----------------------------------------------------------------------------------------------------------------------

class CudaUtils {
private:
    cudaSurfaceObject_t viewCudaSurfaceObject;
public:
    CudaUtils();
    ~CudaUtils();
    void initializeRenderSurface(Texture* texture);
    void renderScene(CudaScene* cudaScene);
    void onClick(int x, int y, CudaScene* cudaScene);
    void deviceInformation();
};

//----------------------------------------------------------------------------------------------------------------------

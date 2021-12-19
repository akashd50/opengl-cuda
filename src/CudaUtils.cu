#pragma once
#include <iostream>
#include "glm/glm.hpp"
#include "headers/CudaUtils.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>

__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ uchar4 toRGBA(const float3 &a) {
    return make_uchar4(int(a.x * 255), int(a.y * 255), int(a.z * 255), 255);
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

__device__ HitInfo doHitTest(float3 eye, float3 ray, CudaScene* scene) {
    HitInfo hit;
    for (int i=0; i<scene->numObjects; i++) {
        CudaSphere* sphere = (CudaSphere*)scene->objects[i];
        float sphereHit = check_hit_on_sphere(eye, ray, sphere->position, sphere->radius);
        if (sphereHit >= 0 && sphereHit != MIN_T) {
            hit.object = sphere;
            hit.t = sphereHit;
            return hit;
        }
    }
    return hit;
}

__global__ void traceRays(cudaSurfaceObject_t image, CudaScene* scene)
{
    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    unsigned int x = threadIdx.x;
    unsigned int y = blockIdx.x;

    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, 512, 512) - eye;
    uchar4 color;
    //printf("IN KERNEL: num_obj(%d) obj_info(%d)", scene->numObjects, scene->objects[0]->type);
    HitInfo hitInfo = doHitTest(eye, ray, scene);
    if (hitInfo.isHit()) {
        color = toRGBA(hitInfo.object->material.ambient);
    } else {
        color = make_uchar4(0, 0, 0, 255);
    }

    surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
}

//----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------Cuda Utils Class Definition----------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

CudaUtils::CudaUtils() {
}

CudaUtils::~CudaUtils() {

}

void CudaUtils::initializeRenderSurface(Texture* texture) {
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

    check(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
}

void CudaUtils::renderScene(CudaScene* cudaScene) {
    traceRays<<<512, 512>>>(CudaUtils::viewCudaSurfaceObject, cudaScene);
}

void CudaUtils::deviceInformation() {
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

        CUdevice device;
        cuDeviceGet(&device, i);
        int major, minor;
        //cuDeviceComputeCapability(&major, &minor, device);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        std::cout << "Minor: " << minor << " \nMajor: " << major << std::endl;
    }

}

//----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------Additional Utilities-----------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

float3 vec3ToFloat3(glm::vec3 vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

CudaMaterial materialToCudaMaterial(Material* material) {
    CudaMaterial newMaterial(vec3ToFloat3(material->ambient), vec3ToFloat3(material->diffuse), vec3ToFloat3(material->specular),
                             material->shininess, vec3ToFloat3(material->reflective), vec3ToFloat3(material->transmissive),
                             material->refraction, material->roughness);
    return newMaterial;
}

CudaRTObject* rtObjectToCudaRTObject(RTObject* object) {
    switch (object->getType()) {
        case SPHERE:
            Sphere* sphere = (Sphere*)object;
            CudaSphere newSphere(vec3ToFloat3(sphere->getPosition()), sphere->getRadius(), materialToCudaMaterial(object->getMaterial()));
            CudaSphere* cudaPointer;
            check(cudaMalloc((void**)&cudaPointer, sizeof(CudaSphere)));
            check(cudaMemcpy(cudaPointer, &newSphere, sizeof(CudaSphere), cudaMemcpyHostToDevice));
            return cudaPointer;
    }
    return nullptr;
}

CudaScene* allocateCudaScene(Scene* scene) {
    int numObjects = scene->getObjects().size();
    auto objects = new CudaRTObject*[numObjects];
    int index = 0;
    for (RTObject* obj : scene->getObjects()) {
        CudaRTObject* cudaPtr = rtObjectToCudaRTObject(obj);
        objects[index++] = cudaPtr;
    }

    CudaRTObject** cudaObjectsPtr;
    check(cudaMalloc((void**)&cudaObjectsPtr, numObjects * sizeof(CudaRTObject*)));
    check(cudaMemcpy(cudaObjectsPtr, objects, numObjects * sizeof(CudaRTObject*), cudaMemcpyHostToDevice));
    CudaScene cudaScene(cudaObjectsPtr, numObjects);
    CudaScene* cudaScenePtr;
    check(cudaMalloc((void**)&cudaScenePtr, sizeof(CudaScene)));
    check(cudaMemcpy(cudaScenePtr, &cudaScene, sizeof(CudaScene), cudaMemcpyHostToDevice));

//    CudaRTObject** objs = new CudaRTObject*[1];
//    check(cudaMemcpy(objs, cudaObjectsPtr, sizeof(CudaRTObject*), cudaMemcpyDeviceToHost))
//
//    CudaSphere* sphere = (CudaSphere*)malloc(sizeof(CudaSphere));
//    check(cudaMemcpy(sphere, objs[0], sizeof(CudaSphere), cudaMemcpyDeviceToHost))
    //cudaMemR
    return cudaScenePtr;
}

void cleanCudaScene(CudaScene* scene) {
    for (int i=0; i<scene->numObjects; i++) {
        cudaFree(scene->objects[i]);
    }
    cudaFree(scene);
}

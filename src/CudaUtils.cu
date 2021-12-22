#pragma once
#include <iostream>
#include "glm/glm.hpp"
#include "headers/CudaUtils.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <vector_functions.h>

//----------OPERATORS---------------------------------------------------------------------------------------------------

__device__ uchar4 operator+(const uchar4 &a, const uchar4 &b) {
    return make_uchar4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.z);
}

__device__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ float3 operator*(const float &a, const float3 &b) {
    return b * a;
}

__device__ float3 operator/(const float3 &a, const float &b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//----------VECTOR--OPERATIONS------------------------------------------------------------------------------------------

__device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ uchar4 toRGBA(const float3 &a) {
    return make_uchar4(int(a.x * 255), int(a.y * 255), int(a.z * 255), 255);
}

__device__ float3 t_to_vec(float3 e, float3 d, float t) {
    return e + (t * d);
}

__device__ float3 normalize(float3 a) {
    float mag = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return make_float3(a.x, a.y, a.z)/mag;
}

//----------RT-FUNCTIONS------------------------------------------------------------------------------------------------

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

__device__ float3 getReflectedRay(float3 e, float3 d, float3 normal) {
    float3 ray_dir = normalize(d);
    return ray_dir - 2.0f * normal * dot(ray_dir, normal);
}

__device__ float3 getSphereNormal(float3 point, CudaSphere* sphere) {
    float3 normal = point - sphere->position;
    return normalize(normal);
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
    return MAX_T;
}

__device__ HitInfo doHitTest(float3 eye, float3 ray, CudaScene* scene) {
    HitInfo hit;
    for (int i=0; i<scene->numObjects; i++) {
        CudaSphere* sphere = (CudaSphere*)scene->objects[i];
        float sphereHit = check_hit_on_sphere(eye, ray, sphere->position, sphere->radius);
        if (sphereHit >= HIT_T_OFFSET && sphereHit < hit.t) {
            hit.object = sphere;
            hit.t = sphereHit;
            hit.hitPoint = t_to_vec(eye, ray, sphereHit);
            hit.index = i;
        }
    }
    return hit;
}

__device__ float3 traceSingleRay(float3 eye, float3 ray, CudaScene* scene, int bounceIndex, bool debug) {
    if (bounceIndex > 1) {
        //printf("Bounce greater than 1 ; %d", bounceIndex);
        return make_float3(0, 0, 0);
    }

    float3 color;
    HitInfo hitInfo = doHitTest(eye, ray, scene);
    if (hitInfo.isHit()) {
        float3 reflectedRay = normalize(getReflectedRay(eye, ray, getSphereNormal(hitInfo.hitPoint, (CudaSphere*)hitInfo.object)));

        if (debug) {
            printf("HitInfo(%d); Hit T(%f) @ (%f, %f, %f) - Reflected(%f, %f, %f)\n", hitInfo.index, hitInfo.t, hitInfo.hitPoint.x, hitInfo.hitPoint.y, hitInfo.hitPoint.z,
                   reflectedRay.x, reflectedRay.y, reflectedRay.z);
        }
        float3 reflectedRayColor = hitInfo.object->material->reflective * traceSingleRay(hitInfo.hitPoint, reflectedRay, scene, bounceIndex + 1, debug);
        color = hitInfo.object->material->diffuse + reflectedRayColor;
    } else {
        color = make_float3(0, 0, 0);
    }

    return color;
}

__global__ void kernel_traceRays(cudaSurfaceObject_t image, CudaScene* scene)
{
    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    unsigned int x = threadIdx.x;
    unsigned int y = blockIdx.x;

    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, 512, 512) - eye;
    uchar4 color = toRGBA(traceSingleRay(eye, ray, scene, 0, false));

    surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
}

__global__ void kernel_traceSingleRay(int x, int y, CudaScene* scene)
{
    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, 512, 512) - eye;
    printf("Ray (%f, %f, %f)\n", ray.x, ray.y, ray.z);
    traceSingleRay(eye, ray, scene, 0, true);
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
    kernel_traceRays<<<512, 512>>>(CudaUtils::viewCudaSurfaceObject, cudaScene);
}

void CudaUtils::onClick(int x, int y, CudaScene* cudaScene) {
    kernel_traceSingleRay<<<1, 1>>>(x, y, cudaScene);
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

template <class T>
T* cudaWrite(T* data, int len) {
    T* cudaPointer;
    check(cudaMalloc((void**)&cudaPointer, sizeof(T) * len));
    check(cudaMemcpy(cudaPointer, data, sizeof(T) * len, cudaMemcpyHostToDevice));
    return cudaPointer;
}

template <class T>
T* cudaRead(T* src, int len) {
//    CudaRTObject** objs = new CudaRTObject*[1];
//    check(cudaMemcpy(objs, cudaObjectsPtr, sizeof(CudaRTObject*), cudaMemcpyDeviceToHost))

    T* hostPointer = (T*)malloc(len * sizeof(T));
    check(cudaMemcpy(hostPointer, src, len * sizeof(T), cudaMemcpyDeviceToHost))
    return hostPointer;
}

CudaMaterial* materialToCudaMaterial(Material* material) {
    CudaMaterial newMaterial(vec3ToFloat3(material->ambient), vec3ToFloat3(material->diffuse), vec3ToFloat3(material->specular),
                             material->shininess, vec3ToFloat3(material->reflective), vec3ToFloat3(material->transmissive),
                             material->refraction, material->roughness);
    return cudaWrite<CudaMaterial>(&newMaterial, 1);
}

CudaRTObject* rtObjectToCudaRTObject(RTObject* object) {
    switch (object->getType()) {
        case SPHERE:
            Sphere* sphere = (Sphere*)object;
            CudaSphere newSphere(vec3ToFloat3(sphere->getPosition()), sphere->getRadius(), materialToCudaMaterial(object->getMaterial()));
            return cudaWrite<CudaSphere>(&newSphere, 1);
    }
    return nullptr;
}

CudaScene* allocateCudaScene(Scene* scene) {
    int numObjects = scene->getObjects().size();
    auto objects = new CudaRTObject*[numObjects];
    int index = 0;
    for (RTObject* obj : scene->getObjects()) {
        CudaRTObject* cudaPtr = rtObjectToCudaRTObject(obj);
        if (cudaPtr != nullptr) {
            objects[index++] = cudaPtr;
        }
    }

    CudaRTObject** cudaObjectsPtr = cudaWrite<CudaRTObject *>(objects, index);
    CudaScene cudaScene(cudaObjectsPtr, index);
    return cudaWrite<CudaScene>(&cudaScene, 1);
}

BVHBinaryNode* createTreeHelper(std::vector<CudaTriangle*>* localTriangles, BVHBinaryNode* node) {
    int len = localTriangles->size();
    if (len <= 5) {
        int* indices = new int[len];
        for (int i=0; i<len; i++) {
            indices[i] = localTriangles->at(i)->index;
        }
        node->objectsIndex = indices;

        BVHBinaryNode tempNode(cudaWrite<Bounds>(node->bounds, 1), cudaWrite<int>(indices, len));
        return cudaWrite<BVHBinaryNode>(&tempNode, 1);
    }

    auto leftTriangles = new std::vector<CudaTriangle*>();
    auto rightTriangles = new std::vector<CudaTriangle*>();

    //bool xDiv, yDiv, zDiv;
    auto nb = *node->bounds;
    float xLen = nb.right - nb.left;
    float yLen = nb.top - nb.bottom;
    float zLen = nb.right - nb.left;
    if (xLen > yLen && xLen > zLen) {
        //xDiv = true;
        float mid = (nb.left + nb.right)/2;
        node->left = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, mid, nb.front, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, mid, nb.right, nb.front, nb.back));
    }
    else if (yLen > xLen && yLen > zLen) {
        //yDiv = true;
        float mid = (nb.top + nb.bottom)/2;
        node->left = new BVHBinaryNode(new Bounds(mid, nb.bottom, nb.left, nb.right, nb.front, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, mid, nb.left, nb.right, nb.front, nb.back));
    }
    else if (zLen > yLen && zLen > xLen) {
        //zDiv = true;
        float mid = (nb.front + nb.back)/2;
        node->left = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, nb.right, mid, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, nb.right, nb.front, mid));
    }

    for (CudaTriangle* t : *localTriangles) {
        //divide along the axis with max length
        if (isTriangleInBounds(t, node->left->bounds)) {
            leftTriangles->push_back(t);
        }
        else if (isTriangleInBounds(t, node->right->bounds)) {
            rightTriangles->push_back(t);
        }
    }

    BVHBinaryNode* leftNode = createTreeHelper(leftTriangles, node->left);
    delete leftTriangles;
    BVHBinaryNode* rightNode = createTreeHelper(rightTriangles, node->right);
    delete rightTriangles;

    BVHBinaryNode tempNode(cudaWrite<Bounds>(node->bounds, 1), leftNode, rightNode);
    return cudaWrite<BVHBinaryNode>(&tempNode, 1);
}

bool isTriangleInBounds(CudaTriangle* triangle, Bounds* bounds) {
    float3 pos = triangle->getPosition();
    return (pos.x > bounds->left && pos.x < bounds->right) &&
           (pos.y > bounds->bottom && pos.y < bounds->top) &&
           (pos.z > bounds->back && pos.z < bounds->front);
}

//    std::vector<CudaTriangle*> trianglesInBounds(std::vector<CudaTriangle*>* localTriangles, Bounds* bounds) {
//        auto leftTriangles = new std::vector<CudaTriangle*>();
//        for (CudaTriangle* t : *localTriangles) {
//
//        }
//    }

//    void createTreeHelper(CudaTriangle** localTriangles, int num, float3 position) {
//        for (int i=0; i<num; i++) {
//            CudaTriangle* t = localTriangles[i];
//            float3 position = t->getPosition();
//            if (position.y >= position.y) {
//                // If in the top half
//                if (position.x >= position.x) {
//                    // If in the top right half
//                    if (position.z >= position.z) {
//                        // If in the top right front
//                    } else {
//                        // If in the top right back
//                    }
//                } else {
//                    // If in the top left half
//                    if (position.z >= position.z) {
//                        // If in the top left front
//
//                    } else {
//                        // If in the top left back
//
//                    }
//                }
//            } else {
//                // If in the bottom half
//                if (position.x >= position.x) {
//                    // If in the bottom right half
//                    if (position.z >= position.z) {
//                        // If in the bottom right front
//                    } else {
//                        // If in the bottom right back
//                    }
//                } else {
//                    // If in the bottom left half
//                    if (position.z >= position.z) {
//                        // If in the bottom left front
//
//                    } else {
//                        // If in the bottom left back
//
//                    }
//                }
//            }
//        }
//    }

void cleanCudaScene(CudaScene* scene) {
    for (int i=0; i<scene->numObjects; i++) {
        cudaFree(scene->objects[i]);
    }
    cudaFree(scene);
}

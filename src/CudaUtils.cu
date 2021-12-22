#pragma once
#include <iostream>
#include "glm/glm.hpp"
#include "headers/CudaUtils.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <vector_functions.h>
#include <math_functions.h>

//----------OPERATORS---------------------------------------------------------------------------------------------------

__device__ __host__ uchar4 operator+(const uchar4 &a, const uchar4 &b) {
    return make_uchar4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.z);
}

__device__ __host__ float3 operator*(const float3 &a, const float &b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ __host__ float3 operator*(const float &a, const float3 &b) {
    return b * a;
}

__device__ __host__ float3 operator/(const float3 &a, const float &b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__device__ __host__ float3 operator/(const float a, const float3 &b) {
    return make_float3(a/b.x, a/b.y, a/b.z);
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __host__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

//----------VECTOR--OPERATIONS------------------------------------------------------------------------------------------

__device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ uchar4 toRGBA(const float3 &a) {
    return make_uchar4(int(a.x * 255), int(a.y * 255), int(a.z * 255), 255);
}

__device__ __host__ float3 t_to_vec(float3 e, float3 d, float t) {
    return e + (t * d);
}

__device__ float magnitude(float3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ float3 normalize(float3 a) {
    float mag = magnitude(a);
    return make_float3(a.x, a.y, a.z)/mag;
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
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

__device__ float checkHitOnPlane(float3 e, float3 d, float3 center, float3 normal) {
    /*Checks the hit on an infinite plane for the given normal and returns t value*/
    float denominator = dot(normal, d);
    if (denominator != 0.0) {
        float t = dot(normal, (center - e)) / denominator;
        return t;
    }
    return MAX_T;
}

__device__ float checkHitOnTriangle(float3 e, float3 d, float3 a, float3 b, float3 c) {
    /*Checks the hit on the triangle and returns t value. I first use the plane hit and then check if its inside the triangle*/
    float3 normal = normalize(cross(b - a, c - a));
    float t = checkHitOnPlane(e, d, a, normal);
    float3 x = t_to_vec(e, d, t);
    float aTest = dot(cross(b - a, x - a), normal);
    float bTest = dot(cross(c - b, x - b), normal);
    float cTest = dot(cross(a - c, x - c), normal);
    if (t != MAX_T && ((aTest >= 0 - HIT_T_OFFSET && bTest >= 0 - HIT_T_OFFSET && cTest >= 0 - HIT_T_OFFSET)
    || (aTest <= 0 + HIT_T_OFFSET && bTest <= 0 + HIT_T_OFFSET && cTest <= 0 + HIT_T_OFFSET))) {
        return t;
    }
    return MAX_T;
}

__device__ void printBounds(Bounds* bounds) {
    printf("AABB (%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f)", bounds->top, bounds->bottom,
           bounds->left, bounds->right, bounds->front, bounds->back);
}

__device__ void print2DUtil(BVHBinaryNode *root, int space)
{
    // Base case
    if (root == nullptr)
        return;

    // Increase distance between levels
    space += 10;

    // Process right child first
    print2DUtil(root->right, space);

    // Print current node after space
    // count
    printf("\n");
    for (int i = 10; i < space; i++)
        printf(" ");
    //print data
    printf("[{");
    //if (root->numObjects == 0) {
    printBounds(root->bounds);
    //}
    printf("} ");
    for (int i=0; i<root->numObjects; i++) {
        printf("%d, ", root->objectsIndex[i]);
    }
    printf("]");
    //cout<<root->data<<"\n";

    // Process left child
    print2DUtil(root->left, space);
}

__device__ __host__ void swap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}

/*
 * float3 invDir = 1.0 / d;
    //float3 invDir = make_float3(0.0, 0.0, 0.0) - d;

    float xVal = (invDir.x < 0) ? bounds->right : bounds->left;
    float yVal = (invDir.y < 0) ? bounds->top : bounds->bottom;
    float zVal = (invDir.z < 0) ? bounds->front : bounds->back;
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin = (xVal - e.x) * invDir.x;
    tmax = (xVal - e.x) * invDir.x;
    tymin = (yVal - e.y) * invDir.y;
    tymax = (yVal - e.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return MAX_T;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (zVal - e.z) * invDir.z;
    tzmax = (zVal - e.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return MAX_T;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
 */

__device__ __host__ float checkHitOnAABB(float3 e, float3 d, Bounds* bounds, bool debug) {
    float tmin = (bounds->left - e.x) / d.x;
    float tmax = (bounds->right - e.x) / d.x;

    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (bounds->bottom - e.y) / d.y;
    float tymax = (bounds->top - e.y) / d.y;

    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return MAX_T;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (bounds->back - e.z) / d.z;
    float tzmax = (bounds->front - e.z) / d.z;

    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return MAX_T;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return tmin;
}

__device__ float checkHitOnMeshHelper(float3 eye, float3 ray, BVHBinaryNode* node, CudaMesh* mesh, bool debug) {
    float minT = MAX_T;
    if (node->numObjects != 0) { // Is a leaf node
        for (int j=0; j<node->numObjects; j++) {
            CudaTriangle t = mesh->triangles[node->objectsIndex[j]];
            float triangleHit = checkHitOnTriangle(eye, ray, t.a, t.b, t.c);
            if(debug) {
                printf("Checking hits on triangle (%d) -- (%f)\n", node->objectsIndex[j], triangleHit);
            }
            if (triangleHit < minT) {
                minT = triangleHit;
                if (debug) {
                    printf("New hit on triangle at (%d) MinT (%f)\n", node->objectsIndex[j], minT);
                }
            }
        }
    }

    if (node->left == nullptr || node->right == nullptr) {
        return minT;
    }

    float leftT = checkHitOnAABB(eye, ray, node->left->bounds, debug);
    float rightT = checkHitOnAABB(eye, ray, node->right->bounds, debug);

//    if (debug) {
//        printf("LeftT - AABB (%f, %f, %f, %f, %f, %f) ---- (%f)\n", node->left->bounds->top, node->left->bounds->bottom,
//               node->left->bounds->left, node->left->bounds->right, node->left->bounds->front, node->left->bounds->back, leftT);
//
//        printf("RightT - AABB (%f, %f, %f, %f, %f, %f) ---- (%f)\n", node->right->bounds->top, node->right->bounds->bottom,
//               node->right->bounds->left, node->right->bounds->right, node->right->bounds->front, node->right->bounds->back, rightT);
//    }

    if (leftT != MAX_T && leftT <= minT) {
        if (debug) {
            printf("Checking left LeftT(%f) MinT(%f)\n", leftT, minT);
        }
        float tempT = checkHitOnMeshHelper(eye, ray, node->left, mesh, debug);
        minT = min(tempT, minT);
    }
    if (rightT != MAX_T && rightT <= minT) {
        if (debug) {
            printf("Checking right RightT(%f) MinT(%f)\n", rightT, minT);
        }
        float tempT = checkHitOnMeshHelper(eye, ray, node->right, mesh, debug);
        minT = min(tempT, minT);
    }

    if (debug) {
        printf("Returning MinT(%f)\n", minT);
    }
    return minT;
}

__device__ float checkHitOnMesh(float3 eye, float3 ray, BVHBinaryNode* node, CudaMesh* mesh, bool debug) {
    float t = checkHitOnAABB(eye, ray, node->bounds, debug);
    if (debug) {
//        printf("\n\n");
//        print2DUtil(node, 0);
//        printf("\n\n");

        printf("Main AABB Hit @ (%f)\n", t);
    }

    if (t != MAX_T) { // If node is hit
        return checkHitOnMeshHelper(eye, ray, node, mesh, debug);
    } else {
        return MAX_T;
    }
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

__device__ HitInfo doHitTest(float3 eye, float3 ray, CudaScene* scene, bool debug) {
    HitInfo hit;
    for (int i=0; i<scene->numObjects; i++) {
        if (scene->objects[i]->type == SPHERE) {
            CudaSphere* sphere = (CudaSphere*)scene->objects[i];
            float sphereHit = check_hit_on_sphere(eye, ray, sphere->position, sphere->radius);
            if (sphereHit >= HIT_T_OFFSET && sphereHit < hit.t) {
                hit.object = sphere;
                hit.t = sphereHit;
                hit.hitPoint = t_to_vec(eye, ray, sphereHit);
                hit.index = i;

                if (debug) {
                    printf("doHitTest @ index (%d) with t (%f)\n", i, sphereHit);
                }
            }
        }
        else if (scene->objects[i]->type == MESH) {
            CudaMesh* mesh = (CudaMesh*)scene->objects[i];
            float meshHit = checkHitOnMesh(eye, ray, mesh->bvhRoot, mesh, debug);
            if (meshHit >= HIT_T_OFFSET && meshHit < hit.t) {
                hit.object = mesh;
                hit.t = meshHit;
                hit.hitPoint = t_to_vec(eye, ray, meshHit);
                hit.index = i;
                if (debug) {
                    printf("doHitTest @ index (%d) with t (%f)\n", i, meshHit);
                }
            }
//            for (int j=0; j<mesh->numTriangles; j++) {
//                CudaTriangle t = mesh->triangles[j];
//                float triangleHit = checkHitOnTriangle(eye, ray, t.a, t.b, t.c);
//                if (triangleHit >= HIT_T_OFFSET && triangleHit < hit.t) {
//                    hit.object = mesh;
//                    hit.t = triangleHit;
//                    hit.hitPoint = t_to_vec(eye, ray, triangleHit);
//                    hit.index = i;
//                }
//            }
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
    HitInfo hitInfo = doHitTest(eye, ray, scene, debug);
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

    surf2Dwrite(color, image, x * sizeof(color), 512-y, cudaBoundaryModeClamp);
}

__global__ void kernel_traceSingleRay(cudaSurfaceObject_t image, int x, int y, CudaScene* scene)
{
    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, 512, 512) - eye;
    printf("Ray (%f, %f, %f)\n", ray.x, ray.y, ray.z);
    uchar4 color = toRGBA(traceSingleRay(eye, ray, scene, 0, true));
    printf("Final Color: (%d, %d, %d, %d)\n", color.x, color.y, color.z, color.w);
    surf2Dwrite(color, image, x * sizeof(color), 512-y, cudaBoundaryModeClamp);
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

    //test hits
//    Bounds* test = new Bounds(0.5, -0.5, -0.5, 0.5, -2.0, -3.0);
//    std::cout << "AABB HIT: " << checkHitOnAABB(make_float3(0.0, 0.0, 0.0), make_float3(0.0, 0.0, -1.0), test) << std::endl;
}

void CudaUtils::onClick(int x, int y, CudaScene* cudaScene) {
    kernel_traceSingleRay<<<1, 1>>>(CudaUtils::viewCudaSurfaceObject, x, y, cudaScene);
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

Bounds* getNewBounds(std::vector<CudaTriangle*>* triangles) {
    auto b = new Bounds();
    for(CudaTriangle* t: *triangles) {
        float maxX = std::fmax(std::fmax(t->a.x, t->b.x), t->c.x);
        float minX = std::fmin(std::fmin(t->a.x, t->b.x), t->c.x);
        float maxY = std::fmax(std::fmax(t->a.y, t->b.y), t->c.y);
        float minY = std::fmin(std::fmin(t->a.y, t->b.y), t->c.y);
        float maxZ = std::fmax(std::fmax(t->a.z, t->b.z), t->c.z);
        float minZ = std::fmin(std::fmin(t->a.z, t->b.z), t->c.z);
        b->right = std::fmax(maxX, b->right);
        b->left = std::fmin(minX, b->left);
        b->top = std::fmax(maxY, b->top);
        b->bottom = std::fmin(minY, b->bottom);
        b->front = std::fmax(maxZ, b->front);
        b->back = std::fmin(minZ, b->back);
    }
    return b;
}

bool isFloat3InBounds(float3 point, Bounds* bounds) {
    return (point.x >= bounds->left && point.x <= bounds->right) &&
           (point.y >= bounds->bottom && point.y <= bounds->top) &&
           (point.z >= bounds->back && point.z <= bounds->front);
}

bool isTriangleInBounds(CudaTriangle* triangle, Bounds* bounds) {
//    float3 pos = triangle->getPosition();
//    return (pos.x >= bounds->left && pos.x <= bounds->right) &&
//           (pos.y >= bounds->bottom && pos.y <= bounds->top) &&
//           (pos.z >= bounds->back && pos.z <= bounds->front);
    return isFloat3InBounds(triangle->a, bounds)
    && isFloat3InBounds(triangle->b, bounds)
    && isFloat3InBounds(triangle->c, bounds);
}

BVHBinaryNode* createTreeHelper(std::vector<CudaTriangle*>* localTriangles, BVHBinaryNode* node) {
    int len = localTriangles->size();
    if (len <= 5) {
        int* indices = new int[len];
        for (int i=0; i<len; i++) {
            indices[i] = localTriangles->at(i)->index;
        }
        node->objectsIndex = indices;

        BVHBinaryNode tempNode(cudaWrite<Bounds>(node->bounds, 1), cudaWrite<int>(indices, len), len);
        return cudaWrite<BVHBinaryNode>(&tempNode, 1);
    }

    auto leftTriangles = new std::vector<CudaTriangle*>();
    auto rightTriangles = new std::vector<CudaTriangle*>();

    //bool xDiv, yDiv, zDiv;
    auto nb = *node->bounds;
    float xLen = nb.right - nb.left;
    float yLen = nb.top - nb.bottom;
    float zLen = nb.right - nb.left;
    if (xLen >= yLen && xLen >= zLen) {
        //xDiv = true;
        float mid = (nb.left + nb.right)/2;
        node->left = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, mid, nb.front, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, mid, nb.right, nb.front, nb.back));
    }
    else if (yLen >= xLen && yLen >= zLen) {
        //yDiv = true;
        float mid = (nb.top + nb.bottom)/2;
        node->left = new BVHBinaryNode(new Bounds(mid, nb.bottom, nb.left, nb.right, nb.front, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, mid, nb.left, nb.right, nb.front, nb.back));
    }
    else if (zLen >= yLen && zLen >= xLen) {
        //zDiv = true;
        float mid = (nb.front + nb.back)/2;
        node->left = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, nb.right, mid, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, nb.right, nb.front, mid));
    }

    std::vector<int> currNodeIndices;
    for (CudaTriangle* t : *localTriangles) {
        //divide along the axis with max length
        if (isTriangleInBounds(t, node->left->bounds)) {
            leftTriangles->push_back(t);
        }
        else if (isTriangleInBounds(t, node->right->bounds)) {
            rightTriangles->push_back(t);
        } else {
            currNodeIndices.push_back(t->index);
        }
    }

    node->left->bounds = getNewBounds(leftTriangles);
    node->right->bounds = getNewBounds(rightTriangles);

    BVHBinaryNode* leftNode = createTreeHelper(leftTriangles, node->left);
    delete leftTriangles;
    BVHBinaryNode* rightNode = createTreeHelper(rightTriangles, node->right);
    delete rightTriangles;

    BVHBinaryNode tempNode(cudaWrite<Bounds>(node->bounds, 1), leftNode, rightNode);
    tempNode.objectsIndex = cudaWrite<int>(currNodeIndices.data(), currNodeIndices.size());
    tempNode.numObjects = currNodeIndices.size();
    return cudaWrite<BVHBinaryNode>(&tempNode, 1);
}

CudaMaterial* materialToCudaMaterial(Material* material) {
    CudaMaterial newMaterial(vec3ToFloat3(material->ambient), vec3ToFloat3(material->diffuse), vec3ToFloat3(material->specular),
                             material->shininess, vec3ToFloat3(material->reflective), vec3ToFloat3(material->transmissive),
                             material->refraction, material->roughness);
    return cudaWrite<CudaMaterial>(&newMaterial, 1);
}

CudaRTObject* rtObjectToCudaRTObject(RTObject* object) {
    switch (object->getType()) {
        case SPHERE: {
            Sphere* sphere = (Sphere*)object;
            CudaSphere newSphere(vec3ToFloat3(sphere->getPosition()), sphere->getRadius(), materialToCudaMaterial(object->getMaterial()));
            return cudaWrite<CudaSphere>(&newSphere, 1);
        }
        case MESH: {
            Mesh* mesh = (Mesh*)object;
            std::vector<CudaTriangle> tempTriangles;
            std::vector<CudaTriangle*> treeTriangles;
            for(int i=0; i<mesh->triangles->size(); i++) {
                Triangle* t = mesh->triangles->at(i);
                CudaTriangle* cudaTriangle = new CudaTriangle(vec3ToFloat3(t->a), vec3ToFloat3(t->b), vec3ToFloat3(t->c), i);
                tempTriangles.push_back(*cudaTriangle);
                treeTriangles.push_back(cudaTriangle);
            }
            CudaTriangle* cudaTrianglePtr = cudaWrite<CudaTriangle>(tempTriangles.data(), tempTriangles.size());
            CudaMesh tempMesh(cudaTrianglePtr);
            tempMesh.numTriangles = tempTriangles.size();
            BVHBinaryNode root(mesh->bounds);
            tempMesh.bvhRoot = createTreeHelper(&treeTriangles, &root);
            tempMesh.material = materialToCudaMaterial(object->getMaterial());
            for (CudaTriangle* t: treeTriangles) {
                delete t;
            }

            return cudaWrite<CudaMesh>(&tempMesh, 1);
        }
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
    delete[] objects;

    CudaScene cudaScene(cudaObjectsPtr, index);
    return cudaWrite<CudaScene>(&cudaScene, 1);
}

//CudaMaterial* materialToCudaMaterialOnHost(Material* material) {
//    CudaMaterial* newMaterial = new CudaMaterial(vec3ToFloat3(material->ambient), vec3ToFloat3(material->diffuse), vec3ToFloat3(material->specular),
//                             material->shininess, vec3ToFloat3(material->reflective), vec3ToFloat3(material->transmissive),
//                             material->refraction, material->roughness);
//    return newMaterial;
//}
//
//CudaRTObject* rtObjectToCudaRTObjectOnHost(RTObject* object) {
//    switch (object->getType()) {
//        case SPHERE: {
//            Sphere* sphere = (Sphere*)object;
//            return new CudaSphere(vec3ToFloat3(sphere->getPosition()), sphere->getRadius(), materialToCudaMaterial(object->getMaterial()));;
//        }
//        case MESH: {
//            Mesh* mesh = (Mesh*)object;
//            std::vector<CudaTriangle> tempTriangles;
//            std::vector<CudaTriangle*> treeTriangles;
//            for(int i=0; i<mesh->triangles->size(); i++) {
//                Triangle* t = mesh->triangles->at(i);
//                CudaTriangle* cudaTriangle = new CudaTriangle(vec3ToFloat3(t->a), vec3ToFloat3(t->b), vec3ToFloat3(t->c), i);
//                tempTriangles.push_back(*cudaTriangle);
//                treeTriangles.push_back(cudaTriangle);
//            }
//            CudaMesh tempMesh(tempTriangles.data());
//            tempMesh.numTriangles = tempTriangles.size();
//            BVHBinaryNode root(mesh->bounds);
//            tempMesh.bvhRoot = createTreeHelper(&treeTriangles, &root);
//            tempMesh.material = materialToCudaMaterial(object->getMaterial());
//            for (CudaTriangle* t: treeTriangles) {
//                delete t;
//            }
//
//            return cudaWrite<CudaMesh>(&tempMesh, 1);
//        }
//}

//CudaScene* allocateCudaSceneOnHost(Scene* scene) {
//    int numObjects = scene->getObjects().size();
//    auto objects = new CudaRTObject*[numObjects];
//    int index = 0;
//    for (RTObject* obj : scene->getObjects()) {
//        CudaRTObject* cudaPtr = rtObjectToCudaRTObject(obj);
//        if (cudaPtr != nullptr) {
//            objects[index++] = cudaPtr;
//        }
//    }
//    CudaRTObject** cudaObjectsPtr = cudaWrite<CudaRTObject *>(objects, index);
//    delete[] objects;
//
//    CudaScene cudaScene(cudaObjectsPtr, index);
//    return cudaWrite<CudaScene>(&cudaScene, 1);
//}


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

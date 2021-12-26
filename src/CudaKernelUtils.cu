#pragma once
#include <iostream>
#include "headers/CudaKernelUtils.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <vector_functions.h>

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

__device__ __host__ float3 operator-(const float a, const float3 &b) {
    return make_float3(a-b.x, a-b.y, a-b.z);
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

__device__ float len_squared(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ float3 normalize(float3 a) {
    float mag = magnitude(a);
    return make_float3(a.x, a.y, a.z)/mag;
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float3 clamp(float3 a, float min, float max) {
    float x = a.x; float y = a.y; float z = a.z;
    x = x < min ? min : x; x = x > max ? max : x;
    y = y < min ? min : y; y = y > max ? max : y;
    z = z < min ? min : z; z = z > max ? max : z;
    return make_float3(x, y, z);
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

__device__ float3 getTriangleNormal(float3 a, float3 b, float3 c) {
    return normalize(cross(b - a, c - a));
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
    float3 normal = getTriangleNormal(a, b, c);
    float t = checkHitOnPlane(e, d, a, normal);
    float3 x = t_to_vec(e, d, t);
    float aTest = dot(cross(b - a, x - a), normal);
    float bTest = dot(cross(c - b, x - b), normal);
    float cTest = dot(cross(a - c, x - c), normal);
    if (t != MAX_T && ((aTest >= 0 - HIT_T_OFFSET_1 && bTest >= 0 - HIT_T_OFFSET_1 && cTest >= 0 - HIT_T_OFFSET_1)
    || (aTest <= 0 + HIT_T_OFFSET_1 && bTest <= 0 + HIT_T_OFFSET_1 && cTest <= 0 + HIT_T_OFFSET_1))) {
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
    //printBounds(root->bounds);
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

__device__ void swap(float &a, float &b) {
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

__device__ MinMaxT checkHitOnAABB(float3 &eye, float3 &ray, Bounds* bounds, bool debug) {
    if (bounds == nullptr) return {MIN_T, MAX_T};

    float tmin = (bounds->left - eye.x) / ray.x;
    float tmax = (bounds->right - eye.x) / ray.x;

    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (bounds->bottom - eye.y) / ray.y;
    float tymax = (bounds->top - eye.y) / ray.y;

    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return {MIN_T, MAX_T};

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (bounds->back - eye.z) / ray.z;
    float tzmax = (bounds->front - eye.z) / ray.z;

    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return {MIN_T, MAX_T};

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return {tmin != tmin ? 0 : tmin, tmax != tmax ? 0 : tmax};
}

__device__ void checkHitOnNodeTriangles(float3 &eye, float3 &ray, BVHBinaryNode* node, CudaMesh* mesh, HitInfo &hitInfo, bool debug) {
//    if (debug) {
//        printf("Starting to check hits on triangles in node NumObjects(%d)\n", node->numObjects);
//    }

    if (node != nullptr && node->numObjects != 0) { // Is a leaf node
        for (int j=0; j<node->numObjects; j++) {
            int objIndex = node->objectsIndex[j];
//            if (debug) {
//                printf("Current triangle index (%d)\n", objIndex);
//            }
            CudaTriangle t = mesh->triangles[objIndex];
            float triangleHit = checkHitOnTriangle(eye, ray, t.a, t.b, t.c);
//            if(debug) {
//                printf("Checking hits on triangle (%d) -- (%f)\n", node->objectsIndex[j], triangleHit);
//            }
            if (triangleHit <= hitInfo.t && triangleHit >= HIT_T_OFFSET_1) {
                hitInfo.t = triangleHit;
                hitInfo.hitNormal = getTriangleNormal(t.a, t.b, t.c);
//                if (debug) {
//                    printf("New hit on triangle at (%d) MinT (%f)\n", node->objectsIndex[j], hitInfo.t);
//                }
            }
        }
    }
//    if (debug) {
//        printf("Finished checking hits on triangles in node NumObjects(%d)\n", node->numObjects);
//    }
}

__device__ HitInfo checkHitOnMeshHelperNR(float3 &eye, float3 &ray, CudaMesh* mesh, bool debug) {
    if (debug) {
        printf("Starting to check hit on Mesh\n");
    }
    HitInfo hitInfo;

    auto stack = (Stack<BVHBinaryNode*>*)malloc(sizeof(Stack<BVHBinaryNode*>));
    if (!stack->init()) return hitInfo;
//    if (debug) {
//        printf("Stack initialized\n");
//    }

    float minAABB = MAX_T;
    // start from the root node (set current node to the root node)
    BVHBinaryNode* curr = mesh->bvhRoot;
//    if (debug) {
//        printf("Curr set IsNull(%d)\n", curr == nullptr);
//        printf("Stack Empty (%d)\n", stack->empty());
//    }
    // if the current node is null and the stack is also empty, we are done
    while (!stack->empty() || curr != nullptr)
    {
        // if the current node exists, push it into the stack (defer it)
        // and move to its left child
        if (curr != nullptr)
        {
//            if (debug) {
//                printf("Curr Bounds IsNull(%d)\n", curr->bounds == nullptr);
//            }
            auto currT = checkHitOnAABB(eye, ray, curr->bounds, debug);
            if (debug) {
                printf("AABB Checking curr MinT(%f) MaxT(%f) HitT(%f)\n", currT.minT, currT.maxT, hitInfo.t);
                printBounds(curr->bounds);
                printf("\n");
            }

            if (currT.minT != MIN_T && currT.maxT != MAX_T
            && currT.minT <= hitInfo.t && currT.maxT >= 0) {
                //minAABB = currT;
                checkHitOnNodeTriangles(eye, ray, curr, mesh, hitInfo, debug);
                stack->push(curr);
                curr = curr->left;
            } else {
                curr = nullptr;
                if (debug) { printf("Setting curr to null StackEmpty(%d)\n", stack->empty()); }
            }
        }
        else {
            if (debug) { printf("Curr is null | Popping off of the stack | StackEmpty(%d)\n", stack->empty()); }
            curr = stack->top();
            stack->pop();

            curr = curr->right;
        }
    }

    stack->clean();
    free(stack);
    return hitInfo;
}

__device__ float doTChecks(float newT, float oldT) {
    return newT > 0 && newT < oldT ? newT : oldT;
}

__device__ float check_hit_on_sphere(float3 &eye, float3 &ray, CudaSphere* sphere, bool debug) {
    float3 center_2_eye = eye - sphere->position;
    float ray_dot_ray = dot(ray, ray);
    float discriminant = pow(dot(ray, center_2_eye), 2) - ray_dot_ray * (dot(center_2_eye, center_2_eye) - pow(sphere->radius, 2));

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

__device__ int wang_hash(int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ float newRandomFloat(CudaRandomGenerator* generator, int &randIndex) {
    if (randIndex < generator->numRand) {
        return generator->randomNumbers[randIndex++];
    } else {
        randIndex = 0;
        return generator->randomNumbers[randIndex++];
    }
}

__device__ void randFloat3(curandState &randState, float3 &vec) {
    vec.x = curand_normal(&randState);
    vec.y = curand_normal(&randState);
    vec.z = curand_normal(&randState);

    vec.x = vec.x > 1.0 ? vec.x - 1.0f : vec.x;
    vec.y = vec.y > 1.0 ? vec.y - 1.0f : vec.y;
    vec.z = vec.z > 1.0 ? vec.z - 1.0f : vec.z;
    vec.x = vec.x < -1.0 ? vec.x + 1.0f : vec.x;
    vec.y = vec.y < -1.0 ? vec.y + 1.0f : vec.y;
    vec.z = vec.z < -1.0 ? vec.z + 1.0f : vec.z;
}

__device__ float3 getNewDiffuseRay(HitInfo &hitInfo, CudaThreadData &threadData) {
    CudaRandomGenerator* generator = threadData.scene->generator;
    float3 p_n =  hitInfo.hitPoint + hitInfo.hitNormal;
//    float3 p = p_n + clamp(normalize(make_float3(newRandomFloat(generator, threadData.randIndex), newRandomFloat(generator, threadData.randIndex),
//                                 newRandomFloat(generator, threadData.randIndex))), -hitInfo.object->material->roughness, hitInfo.object->material->roughness);
    float3 vec = make_float3(0, 0, 0);
    while(true) {
        randFloat3(threadData.randState, vec);
        if (len_squared(vec) <= hitInfo.object->material->roughness) break;
    }
    float3 p = p_n + vec;
    return normalize(p - hitInfo.hitPoint);
}

__device__ float3 ray_color(const float3& r) {
    float t = 0.5f * (r.y + 1.0f);
    return (1.0f - t) * make_float3(0.5, 0.7, 1.0) + t * make_float3(1.0, 1.0, 1.0);
}

__device__ HitInfo doHitTest(float3 &eye, float3 &ray, CudaThreadData &threadData) {
    HitInfo hit;
    CudaScene* scene = threadData.scene;
    for (int i=0; i<scene->numObjects; i++) {
        if (scene->objects[i]->type == SPHERE) {
            auto sphere = (CudaSphere*)scene->objects[i];
            float sphereHit = check_hit_on_sphere(eye, ray, sphere, threadData.debug);
            if (sphereHit >= HIT_T_OFFSET && sphereHit < hit.t) {
                hit.object = sphere;
                hit.t = sphereHit;
                hit.hitPoint = t_to_vec(eye, ray, sphereHit);
                hit.hitNormal = getSphereNormal(hit.hitPoint, sphere);
                hit.color = sphere->material->diffuse;
                hit.index = i;
                if (threadData.debug) {
                    printf("doHitTest @ index (%d) with t (%f)\n", i, sphereHit);
                }
            }
        }
        else if (scene->objects[i]->type == MESH) {
            auto mesh = (CudaMesh*)scene->objects[i];
            HitInfo meshHit = checkHitOnMeshHelperNR(eye, ray, mesh, threadData.debug);
            if (meshHit.t >= HIT_T_OFFSET && meshHit.t < hit.t) {
                hit.object = mesh;
                hit.t = meshHit.t;
                hit.hitPoint = t_to_vec(eye, ray, meshHit.t);
                hit.hitNormal = meshHit.hitNormal;
                hit.color = mesh->material->diffuse;
                hit.index = i;
                if (threadData.debug) {
                    printf("doHitTest @ index (%d) with t (%f)\n", i, meshHit.t);
                }
            }
//            for (int j=0; j<mesh->numTriangles; j++) {
//                CudaTriangle t = mesh->triangles[j];
//                float triangleHit = checkHitOnTriangle(eye, ray, t.a, t.b, t.c);
//                if (triangleHit >= HIT_T_OFFSET && triangleHit < hit.t) {
//                    hit.object = mesh;
//                    hit.t = triangleHit;
//                    hit.hitPoint = t_to_vec(eye, ray, triangleHit);
//                    hit.hitNormal = getTriangleNormal(t.a, t.b, t.c);
//                    hit.index = i;
//                }
//            }
        }
    }

    for (int i=0; i<scene->numLights; i++) {
        auto light = (CudaLight*)scene->lights[i];
        if (light->lightType == SKYBOX_LIGHT) {
            auto sphere = ((CudaSkyboxLight*)scene->lights[i])->sphere;
            float sphereHit = check_hit_on_sphere(eye, ray, sphere, threadData.debug);
            if (sphereHit >= HIT_T_OFFSET && sphereHit < hit.t) {
                hit.object = light;
                hit.t = sphereHit;
                hit.hitPoint = t_to_vec(eye, ray, sphereHit);
                hit.hitNormal = getSphereNormal(hit.hitPoint, sphere);
                hit.color = ray_color(ray);
                hit.index = i;
                if (threadData.debug) {
                    printf("doHitTest @ index (%d) with t (%f)\n", i, sphereHit);
                }
            }
        }
    }
    return hit;
}

//__device__ HitInfo bounceDiffuseRays(HitInfo &hitInfo, CudaScene* scene, int &randIndex, int bounceIndex) {
//    HitInfo newHit;
//    float3 col = hitInfo.object->material->diffuse;
//    newHit.color = make_float3(col.x, col.y, col.z);
//    int numRaySamples = 4;
//    for(int n=0; n<numRaySamples; n++) {
//        float3 reflected = getNewDiffuseRay(scene, hitInfo, randIndex);
//        HitInfo t = doHitTest(hitInfo.hitPoint, reflected, scene, false);
//        if (t.object->type == LIGHT && ((CudaLight*)t.object)->lightType == SKYBOX_LIGHT) {
//            newHit.color = newHit.color + 0.5 * ray_color(reflected);
//        }
//    }
//    newHit.color = (newHit.color/(float)numRaySamples);
//    return newHit;
//}

__device__ float3 calculateLighting(HitInfo &hitInfo, CudaThreadData &threadData) {
    if (hitInfo.object->type == LIGHT) {
        return hitInfo.color;
    }
    CudaScene* scene = threadData.scene;
    float3 lighting = make_float3(0, 0, 0);
    for (int i=0; i<scene->numLights; i++) {
        auto light = (CudaLight*)scene->lights[i];
        if (light->lightType == SKYBOX_LIGHT) {
            float3 col = hitInfo.object->material->diffuse;
            float3 tempColor = make_float3(col.x, col.y, col.z);
            int numRaySamples = 8;
            for(int n=0; n<numRaySamples; n++) {
                float3 reflected = getNewDiffuseRay(hitInfo, threadData);
                HitInfo newHit = doHitTest(hitInfo.hitPoint, reflected, threadData);
                if (newHit.object->type == LIGHT && newHit.index == i) {
                    tempColor = tempColor + 0.5 * ray_color(reflected);
                }
            }
            lighting = lighting + (tempColor/(float)numRaySamples);
        }
    }
    return lighting;
}

__device__ float3 traceSingleRay(float3 eye, float3 ray, int maxBounces, CudaThreadData &threadData) {
    auto stack = (Stack<HitInfo>*)malloc(sizeof(Stack<HitInfo>));
    stack->init();

    float3 newRay = ray;
    float3 newEye = eye;
    bool isHit = true;
    int bounceIndex = 0;
    while(bounceIndex < maxBounces && isHit) {
        HitInfo hitInfo = doHitTest(newEye, newRay, threadData);
        if (hitInfo.isHit()) {
            stack->push(hitInfo);
            if (hitInfo.object->type >= LIGHT) {
                // skybox obj
                isHit = false;
                break;
            }

            //newRay = normalize(getReflectedRay(eye, ray, hitInfo.hitNormal));
            newRay = getNewDiffuseRay(hitInfo, threadData);
            newEye = hitInfo.hitPoint;
            if (threadData.debug) {
                printf("HitInfo(%d); Hit T(%f) @ (%f, %f, %f) | Normal(%f, %f, %f) | Reflected(%f, %f, %f)\n",
                       hitInfo.index, hitInfo.t, hitInfo.hitPoint.x, hitInfo.hitPoint.y, hitInfo.hitPoint.z,
                       hitInfo.hitNormal.x, hitInfo.hitNormal.y, hitInfo.hitNormal.z,
                       newRay.x, newRay.y, newRay.z);
            }
        } else {
            isHit = false;
        }
        bounceIndex++;
    }

    // Sum all colors from stack
    float3 color = make_float3(0.0, 0.0, 0.0);
    if (stack->size() >= 2) {
        while(!stack->empty()) {
            HitInfo curr = stack->top();
            stack->pop();
//            if (debug) {
//                printf("\n\nCurr | Index(%d)\n", curr.index);
//                printf("Next | Index(%d)\n", curr.index);
//                printf("Color(%f, %f, %f)\n", color.x, color.y, color.z);
//            }
            if (curr.object->type < LIGHT) {
                //color = curr.color + (curr.object->material->reflective * color);
                float3 ref = curr.object->material->reflective;
                float3 currLighting = calculateLighting(curr, threadData);
                color = (1.0 - ref) * currLighting + (ref * color);
            } else {
                color = curr.color;
            }
        }
    } else if (stack->size() == 1) {
        HitInfo curr = stack->top(); stack->pop();
        color = color + curr.color;
    }

    stack->clean();
    free(stack);

    return color;
}

__global__ void kernel_traceRays(cudaSurfaceObject_t image, CudaScene* scene,  int startRowIndex, int startColIndex)
{
    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    int x = startColIndex + (int)threadIdx.x;
    int y = startRowIndex + (int)blockIdx.x;
    int randIndex = wang_hash(x * y) % scene->width;

    curandState state;
    curand_init (x*y, 0, 0, &state);

    CudaThreadData threadData;
    threadData.debug = false;
    threadData.randState = state;
    threadData.randIndex = randIndex;
    threadData.scene = scene;

    int maxBounces = 4;
    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, scene->width, scene->height) - eye;

    int numSamples = 4;
    float3 sampledColor = make_float3(0, 0, 0);
    float p = 0.0005;
    sampledColor = sampledColor + traceSingleRay(eye, ray + make_float3(p, p, 0), maxBounces, threadData);
    sampledColor = sampledColor + traceSingleRay(eye, ray + make_float3(-p, -p, 0), maxBounces, threadData);
    sampledColor = sampledColor + traceSingleRay(eye, ray + make_float3(-p, p, 0), maxBounces, threadData);
    sampledColor = sampledColor + traceSingleRay(eye, ray + make_float3(p, -p, 0), maxBounces, threadData);
    uchar4 color = toRGBA(sampledColor/(float)numSamples);

//    float3 cVal = traceSingleRay(eye, ray, scene, maxBounces, randIndex, false);
//    uchar4 color = toRGBA(cVal);

    surf2Dwrite(color, image, x * sizeof(uchar4), scene->height - y, cudaBoundaryModeClamp);
}

__global__ void kernel_traceSingleRay(cudaSurfaceObject_t image, int x, int y, CudaScene* scene)
{

//    printf("New Rand %f\n", curand_normal(&s));
//    printf("New Rand %f\n", curand_normal(&s));
//    printf("New Rand %f\n", curand_normal(&s));

    int randIndex = (x * y) % scene->width;
    int maxBounces = 8;

    CudaThreadData threadData;
    curandState state;
    curand_init (x*y, 0, 0, &state);
    threadData.debug = true;
    threadData.randState = state;
    threadData.randIndex = randIndex;
    threadData.scene = scene;

//    for (int i=0; i<20; i++) {
//        float3 vec = randFloat3(threadData.randState);
//        vec.x = vec.x > 1.0 ? vec.x - 1.0f : vec.x;
//        vec.y = vec.y > 1.0 ? vec.y - 1.0f : vec.y;
//        vec.z = vec.z > 1.0 ? vec.z - 1.0f : vec.z;
//
//        vec.x = vec.x < -1.0 ? vec.x + 1.0f : vec.x;
//        vec.y = vec.y < -1.0 ? vec.y + 1.0f : vec.y;
//        vec.z = vec.z < -1.0 ? vec.z + 1.0f : vec.z;
//
//        printf("New Vector (%f, %f, %f)\n", vec.x, vec.y, vec.z);
//        printf("Len Squared %f\n", len_squared(vec));
//    }

//    float3 eye = make_float3(0.0, 0.0, 0.0);
//    float3 ray = cast_ray(x, y, scene->width, scene->height) - eye;
//    printf("\n\nRay (%f, %f, %f)\n", ray.x, ray.y, ray.z);
//    uchar4 color = toRGBA(traceSingleRay(eye, ray, maxBounces, threadData));
//    printf("Final Color: (%d, %d, %d, %d)\n", color.x, color.y, color.z, color.w);
//    surf2Dwrite(color, image, x * sizeof(uchar4), scene->height - y, cudaBoundaryModeClamp);
}

//----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------Cuda Utils Class Definition----------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

CudaKernelUtils::CudaKernelUtils() {}
CudaKernelUtils::~CudaKernelUtils() {}

void CudaKernelUtils::initializeRenderSurface(Texture* texture) {
    size_t stackLimit = 2048;
    cudaDeviceSetLimit(cudaLimitStackSize, stackLimit);
    size_t newHeapLimit = 16777216;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeapLimit);
//    size_t getLimit;
//    cudaDeviceGetLimit(&getLimit, cudaLimitStackSize);
//    std::cout << "New Stack Size: " << getLimit << std::endl;



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

//    curandGenerator_t  randomGenerator;
//    curandCreateGenerator(&randomGenerator, CURAND_RNG_QUASI_SOBOL32);
//    curandSetPseudoRandomGeneratorSeed(randomGenerator, 1);

}

void CudaKernelUtils::renderScene(CudaScene* cudaScene, int blockSize, int numThreads, int startRowIndex, int startColIndex) {
    kernel_traceRays<<<blockSize, numThreads>>>(CudaKernelUtils::viewCudaSurfaceObject, cudaScene, startRowIndex, startColIndex);
    check(cudaDeviceSynchronize());
    //test hits
//    Bounds* test = new Bounds(0.5, -0.5, -0.5, 0.5, -2.0, -3.0);
//    std::cout << "AABB HIT: " << checkHitOnAABB(make_float3(0.0, 0.0, 0.0), make_float3(0.0, 0.0, -1.0), test) << std::endl;
}

void CudaKernelUtils::onClick(int x, int y, CudaScene* cudaScene) {
    kernel_traceSingleRay<<<1, 1>>>(CudaKernelUtils::viewCudaSurfaceObject, x, y, cudaScene);
    check(cudaDeviceSynchronize());
}

void CudaKernelUtils::deviceInformation() {
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

        size_t stackLimit;
        cudaDeviceGetLimit(&stackLimit, cudaLimitStackSize);
        std::cout << "Stack Size: " << stackLimit << std::endl;

        size_t heapLimit;
        cudaDeviceGetLimit(&heapLimit, cudaLimitMallocHeapSize);
        std::cout << "Heap Size: " << heapLimit << std::endl;
    }
}


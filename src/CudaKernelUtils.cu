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

__device__ __host__ uchar4 operator*(const uchar4 &a, const float &b) {
    return make_uchar4(min(int(a.x*b), 255), min(int(a.y*b), 255), min(int(a.z*b), 255), a.w);
}

__device__ __host__ uchar4 operator/(const uchar4 &a, const float b) {
    return make_uchar4(int(a.x/b), int(a.y/b), int(a.z/b), 255);
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

__device__ __host__ void add(float3 &a, float3 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

__device__ __host__ void subtract(float3 &a, float3 &b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

__device__ __host__ void multiply(float3 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

//----------VECTOR--OPERATIONS------------------------------------------------------------------------------------------

__device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ uchar4 toRGBA(const float3 &a) {
    return make_uchar4(int(a.x * 255), int(a.y * 255), int(a.z * 255), 255);
}

__device__ __host__ float3 t_to_vec(Ray &ray, float &t) {
    return ray.origin + (t * ray.direction);
}

__device__ float magnitude(float3 &a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ float len_squared(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ float3 normalize(float3 a) {
    float mag = magnitude(a);
    return make_float3(a.x, a.y, a.z)/mag;
}

__device__ void normalizeR(float3 &a) {
    float mag = magnitude(a);
    a.x = a.x/mag; a.y = a.y/mag; a.z = a.z/mag;
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

__device__ float3 cast_ray(int &x, int &y, int &width, int &height) {
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

__device__ float3 getReflectedRay(float3 &direction, float3 &normal) {
    float3 ray_dir = normalize(direction);
    return ray_dir - 2.0f * normal * dot(ray_dir, normal);
}

__device__ float3 getSphereNormal(float3 &point, CudaSphere* sphere) {
    float3 normal = point - sphere->position;
    return normalize(normal);
}

__device__ float3 getTriangleNormal(CudaTriangle &triangle) {
    return normalize(cross(triangle.b - triangle.a, triangle.c - triangle.a));
}

__device__ float checkHitOnPlane(float3 &e, float3 &d, float3 &center, float3 &normal) {
    /*Checks the hit on an infinite plane for the given normal and returns t value*/
    float denominator = dot(normal, d);
    if (denominator != 0.0) {
        float t = dot(normal, (center - e)) / denominator;
        return t;
    }
    return MAX_T;
}

__device__ bool checkHitOnTriangle(Ray &ray, CudaTriangle &triangle, HitInfo &hitInfo) {
    /*Checks the hit on the triangle and returns t value. I first use the plane hit and then check if its inside the triangle*/
    float3 normal = getTriangleNormal(triangle);
    float t = checkHitOnPlane(ray.origin, ray.direction, triangle.a, normal);

    if (t == MAX_T) return false;

    float3 x = t_to_vec(ray, t);
    float aTest = dot(cross(triangle.b - triangle.a, x - triangle.a), normal);
    float bTest = dot(cross(triangle.c - triangle.b, x - triangle.b), normal);
    float cTest = dot(cross(triangle.a - triangle.c, x - triangle.c), normal);
    bool isValidHit = ((aTest >= 0 - HIT_T_OFFSET_1 && bTest >= 0 - HIT_T_OFFSET_1 && cTest >= 0 - HIT_T_OFFSET_1)
                   || (aTest <= 0 + HIT_T_OFFSET_1 && bTest <= 0 + HIT_T_OFFSET_1 && cTest <= 0 + HIT_T_OFFSET_1));
    if (isValidHit && t >= HIT_T_OFFSET && t < hitInfo.t) {
        hitInfo.t = t;
        hitInfo.point = t_to_vec(ray, hitInfo.t);
        hitInfo.normal = normal;
        return true;
    }
    return false;
}

__device__ bool check_hit_on_sphere(Ray &ray, CudaSphere* sphere, HitInfo &hitInfo, bool debug) {
    float3 center_2_eye = ray.origin - sphere->position;
    float ray_dot_ray = dot(ray.direction, ray.direction);
    float ray_dot_c2e = dot(ray.direction, center_2_eye);
    float discriminant = pow(ray_dot_c2e, 2) - ray_dot_ray * (dot(center_2_eye, center_2_eye) - pow(sphere->radius, 2));
    float mint = MAX_T;
    if (discriminant > 0) {
        discriminant = sqrt(discriminant);
        float init = -ray_dot_c2e;
        float t1 = (init + discriminant) / ray_dot_ray;
        float t2 = (init - discriminant) / ray_dot_ray;

        mint = min(t1, t2);
        if (mint < HIT_T_OFFSET) {
            mint = max(t1, t2);
        }
    }
    else if (discriminant == 0) {
        float init = -ray_dot_c2e;
        mint = init / ray_dot_ray;
    }

    if (mint >= HIT_T_OFFSET && mint < hitInfo.t) {
        hitInfo.t = mint;
        hitInfo.object = sphere;
        hitInfo.point = t_to_vec(ray, hitInfo.t);
        hitInfo.normal = getSphereNormal(hitInfo.point, sphere);
        hitInfo.objectId = sphere->id;
        return true;
    }
    return false;
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

__device__ MinMaxT checkHitOnAABB(Ray &ray, Bounds* bounds, bool debug) {
    if (bounds == nullptr) return {MIN_T, MAX_T};

    float tmin = (bounds->left - ray.origin.x) / ray.direction.x;
    float tmax = (bounds->right - ray.origin.x) / ray.direction.x;

    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (bounds->bottom - ray.origin.y) / ray.direction.y;
    float tymax = (bounds->top - ray.origin.y) / ray.direction.y;

    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return {MIN_T, MAX_T};

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (bounds->back - ray.origin.z) / ray.direction.z;
    float tzmax = (bounds->front - ray.origin.z) / ray.direction.z;

    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return {MIN_T, MAX_T};

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return {tmin != tmin ? 0 : tmin, tmax != tmax ? 0 : tmax};
}

__device__ bool checkHitOnNodeTriangles(Ray &ray, BVHBinaryNode* node, CudaMesh* mesh, HitInfo &hitInfo, bool debug) {
    bool tUpdated = false;
    if (node != nullptr && node->numObjects != 0) { // Is a leaf node
        for (int j=0; j<node->numObjects; j++) {
            int objIndex = node->objectsIndex[j];
            CudaTriangle triangle = mesh->triangles[objIndex];
            if (checkHitOnTriangle(ray, triangle, hitInfo)) {
                tUpdated = true;
                if (debug) {
                    printf("New hit on triangle at (%d) MinT (%f)\n", node->objectsIndex[j], hitInfo.t);
                }
            }
        }
    }

    return tUpdated;
}

__device__ unsigned long numMeshHitChecks = 0;
__device__ unsigned long numOverallTriangleChecks = 0;

__device__ bool checkHitOnMeshHelperNR(Ray &ray, CudaMesh* mesh, HitInfo &hitInfo, bool debug) {
    if (debug) {
        printf("\n\nStarting to check hit on Mesh\n");
    }

    int numTrianglesChecked = 0;
    int numAABBChecks = 0;
    numMeshHitChecks++;

    auto stack = new Stack<BVHBinaryNode*>();
    if (!stack->init()) return false;

    BVHBinaryNode* curr = mesh->bvhRoot;

    bool tUpdated = false;
    while (!stack->empty() || curr != nullptr) {
        if (curr != nullptr) {
            auto currT = checkHitOnAABB(ray, curr->bounds, debug);
            numAABBChecks++;
            /*if (debug) {
                printf("AABB Checking curr MinT(%f) MaxT(%f) HitT(%f)\n", currT.minT, currT.maxT, hitInfo.t);
                printBounds(curr->bounds);
                printf("\n");
            }*/

            if (currT.minT != MIN_T && currT.maxT != MAX_T
            && currT.minT <= hitInfo.t && currT.maxT >= 0) {
                numTrianglesChecked += curr->numObjects;
                numOverallTriangleChecks += curr->numObjects;
                if (checkHitOnNodeTriangles(ray, curr, mesh, hitInfo, debug)) {
                    tUpdated = true;
                    hitInfo.object = mesh;
                    hitInfo.objectId = mesh->id;
                }

                stack->push(curr);
                curr = curr->left;
            } else {
                curr = nullptr;
            }
        } else {
            curr = stack->top();
            stack->pop();
            curr = curr->right;
        }
    }
    if (debug) printf("\nNum Hit Checked: Triangles(%d) | AABB(%d)\n", numTrianglesChecked, numAABBChecks);

    stack->clean();
    free(stack);
    return tUpdated;
}

__device__ int wang_hash(int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ void randFloat3(curandState &randState, float3 &vec) {
    vec.x = curand_normal(&randState);
    vec.y = curand_normal(&randState);
    vec.z = curand_normal(&randState);
}

__device__ void randFloat3InMesh(curandState &randState, CudaMesh* mesh, float3 &vec) {
    randFloat3(randState, vec);
    normalizeR(vec);
    vec.x = mesh->position.x + vec.x * mesh->dimensions.x;
    vec.y = mesh->position.y + vec.y * mesh->dimensions.y;
    vec.z = mesh->position.z + vec.z * mesh->dimensions.z;
//    vec.x = mesh->position.x + vec.x;
//    vec.y = mesh->position.y + vec.y;
//    vec.z = mesh->position.z + vec.z;
}

__device__ float3 getReflectedDiffuseRay(HitInfo &hitInfo, CudaThreadData &threadData, bool useReflected) {
    float3 p_n = hitInfo.point + (useReflected ? hitInfo.reflected * 1.5 : hitInfo.normal * 1.5);
    float3 vec = make_float3(0, 0, 0);
    randFloat3(threadData.randState, vec);

    normalizeR(vec);
    if (useReflected) {
        vec = vec * hitInfo.object->material->roughness;
    }

    float3 p = p_n + vec;
    return normalize(p - hitInfo.point);
}

__device__ float3 ray_color(const Ray& ray) {
    float t = 0.5f * (ray.direction.y + 1.0f);
    return (1.0f - t) * make_float3(0.5, 0.7, 1.0) + t * make_float3(1.0, 1.0, 1.0);
}

__device__ void doHitTest(Ray &ray, HitInfo &hitInfo, CudaThreadData &threadData) {
    CudaScene* scene = threadData.scene;
    for (int i=0; i<scene->numObjects; i++) {
        if (scene->objects[i]->type == SPHERE) {
            auto sphere = (CudaSphere*)scene->objects[i];
            bool isHit = check_hit_on_sphere(ray, sphere, hitInfo, threadData.debug);
            if (isHit && threadData.debug) {
                printf("\n\ndoHitTest @ index (%d) with t (%f)\n", i, hitInfo.t);
            }
        }
        else if (scene->objects[i]->type == MESH) {
            auto mesh = (CudaMesh*)scene->objects[i];
            bool isHit = checkHitOnMeshHelperNR(ray, mesh, hitInfo, threadData.debug);
            if (isHit && threadData.debug) {
                printf("\n\ndoHitTest @ index (%d) with t (%f)\n", i, hitInfo.t);
            }
        }
    }

    for (int i=0; i<scene->numLights; i++) {
        auto light = (CudaLight*)scene->lights[i];
        if (light->lightType == SKYBOX_LIGHT) {
            auto sphere = ((CudaSkyboxLight*)scene->lights[i])->sphere;
            if (check_hit_on_sphere(ray, sphere, hitInfo, threadData.debug)) {
                hitInfo.object = light;
                hitInfo.lighting = ray_color(ray);
                hitInfo.objectId = light->id;
                if (threadData.debug) {
                    printf("\n\ndoHitTest @ index (%d) with t (%f)\n", i, hitInfo.t);
                }
            }
        } else if (light->lightType == MESH_LIGHT) {
            auto mesh = ((CudaMeshLight*)scene->lights[i])->mesh;
            if (checkHitOnMeshHelperNR(ray, mesh, hitInfo, threadData.debug)) {
                hitInfo.object = light;
                hitInfo.lighting = ((CudaMeshLight*)scene->lights[i])->color;
                hitInfo.objectId = light->id;
                if (threadData.debug) {
                    printf("\n\ndoHitTest @ index (%d) with t (%f)\n", i, hitInfo.t);
                }
            }
        }
    }
}

__device__ void getLighting(HitInfo &hitInfo, CudaThreadData &threadData) {
    //    To Be Used Later when more lights are added

    CudaScene* scene = threadData.scene;
    HitInfo tempHitInfo;
    int numSamplesPerLight = 1;
    for (int i=0; i<scene->numLights; i++) {
        auto light = (CudaLight *) scene->lights[i];
        if (light->lightType == SKYBOX_LIGHT) {
            for (int s=0; s<numSamplesPerLight; s++) {
                float3 randVecToLight = getReflectedDiffuseRay(hitInfo, threadData, false);
                tempHitInfo.t = MAX_T;

                Ray newRay(hitInfo.point, randVecToLight);
                doHitTest(newRay, tempHitInfo, threadData);
                if (tempHitInfo.objectId == light->id) {
                    hitInfo.lighting = hitInfo.lighting + ray_color(newRay) * light->intensity;
                }
            }
        }
        // Maybe only do lighting here and not material stuff - so add them separately in the main trace ray func..
        if (light->lightType == MESH_LIGHT) {
            CudaMeshLight* meshLight = (CudaMeshLight*)light;
            float3 randVecToLight = make_float3(0, 0, 0);
            for (int s=0; s<numSamplesPerLight; s++) {
                randFloat3InMesh(threadData.randState, meshLight->mesh, randVecToLight);
                randVecToLight = normalize(randVecToLight - hitInfo.point);
                tempHitInfo.t = MAX_T;

                Ray newRay(hitInfo.point, randVecToLight);
                doHitTest(newRay, tempHitInfo, threadData);
                if (tempHitInfo.objectId == light->id) {
                    float diff = max(dot(hitInfo.normal, randVecToLight), 0.0f);
                    hitInfo.lighting = hitInfo.lighting + diff * meshLight->color * meshLight->intensity;

//                    float3 viewDir = normalize(viewPos - FragPos);
//                    vec3 reflectDir = reflect(-lightDir, norm);
                }
            }
        }
    }
    hitInfo.lighting = hitInfo.lighting/(float)(numSamplesPerLight * scene->numLights);
}

__device__ void doPathTracing(Ray &ray, HitInfo &hitInfo, CudaThreadData &threadData, float3 &lighting, int &index) {
    if (index >= 3) return;

    if (index != 0) {
        hitInfo.reflected = normalize(getReflectedRay(ray.direction, hitInfo.normal));
        ray.direction = getReflectedDiffuseRay(hitInfo, threadData, true);
    } else {
        ray.direction = getReflectedDiffuseRay(hitInfo, threadData, false);
    }

    hitInfo.t = MAX_T;

    ray.origin = hitInfo.point;
    doHitTest(ray, hitInfo, threadData);
    if (hitInfo.object->type == LIGHT) {
        lighting = lighting + hitInfo.lighting;
    } else {
        index++;
        CudaMaterial* mat = hitInfo.object->material;
        doPathTracing(ray, hitInfo, threadData, lighting, index);
        //Calculate the lighting from all other lights here
        //getLighting(ray, hitInfo, threadData, lighting);
        lighting = lighting * mat->albedo * mat->diffuse;
    }
}

__device__ void calculateLightingPathTraced(HitInfo &hitInfo, CudaThreadData &threadData) {
    if (hitInfo.object->type == LIGHT) {
        // May change this to include other light properties like intensity and stuff
        return;
    }

    float3 lighting = make_float3(0, 0, 0);
    Ray emptyRay(make_float3(0, 0, 0), make_float3(0, 0, 0));
    HitInfo newHit(hitInfo);
    int index = 0;
    doPathTracing(emptyRay, newHit, threadData, lighting, index);
    hitInfo.lighting = lighting * hitInfo.object->material->diffuse;
}

__device__ float3 traceSingleRay(Ray &ray, int maxBounces, CudaThreadData &threadData) {
    auto stack = new Stack<HitInfo>();
    stack->init();

//    float3 newRay = ray.direction;
//    float3 newEye = eye;
    Ray currRay = ray;
    int bounceIndex = 0;
    while(bounceIndex < maxBounces) {
        HitInfo hitInfo;
        doHitTest(currRay, hitInfo, threadData);
        if (hitInfo.isHit()) {
            stack->push(hitInfo);
            if (hitInfo.object->type >= LIGHT) {
                break;
            }

            hitInfo.reflected = normalize(getReflectedRay(ray.direction, hitInfo.normal));
            currRay = Ray(hitInfo.point, getReflectedDiffuseRay(hitInfo, threadData, true));
            if (threadData.debug) {
                printf("\n\nHitInfo(%d); Hit T(%f) @ (%f, %f, %f) | Normal(%f, %f, %f) | Reflected(%f, %f, %f)\n\n\n\n\n\n\n",
                       hitInfo.objectId, hitInfo.t, hitInfo.point.x, hitInfo.point.y, hitInfo.point.z,
                       hitInfo.normal.x, hitInfo.normal.y, hitInfo.normal.z,
                       currRay.direction.x, currRay.direction.y, currRay.direction.z);
            }
        } else {
            break;
        }
        bounceIndex++;
    }

    if (threadData.debug) printf("\n\n\n\n\n\n\n\nStarting to Sum lighting\n\n");
    // Sum all colors from stack
    float3 color = make_float3(0.0, 0.0, 0.0);
    float3 currLighting = make_float3(0, 0, 0);
    if (stack->size() >= 2) {
        while(!stack->empty()) {
            HitInfo curr = stack->top();
            stack->pop();
//            if (debug) {
//                printf("\n\nCurr | Index(%d)\n", curr.index);
//                printf("Next | Index(%d)\n", curr.index);
//                printf("Color(%f, %f, %f)\n", lighting.x, lighting.y, lighting.z);
//            }
            if (curr.object->type < LIGHT) {
                //float3 currLighting = calculateLightingPathTraced(curr, threadData);
                getLighting(curr, threadData);
                currLighting = (currLighting + curr.lighting) * curr.object->material->diffuse;
                color = currLighting + (curr.object->material->reflective * color);
                currLighting = currLighting * curr.object->material->albedo;
            } else {
                //currLighting = curr.lighting;
                color = curr.lighting;
            }
        }
    } else if (stack->size() == 1) {
        HitInfo curr = stack->top(); stack->pop();
        //lighting = lighting + calculateLightingPathTraced(curr, threadData);
        color = curr.lighting;
    }

    color.x = sqrt(color.x);
    color.y = sqrt(color.y);
    color.z = sqrt(color.z);
    color = clamp(color, 0.0, 0.99);

    stack->clean();
    free(stack);

    return color;
}

__global__ void kernel_traceRays(cudaSurfaceObject_t image, CudaScene* scene,  int startRowIndex, int startColIndex, int sampleIndex)
{
    // blockIdx - index of block in grid
    // theadIdx - index of thread in block
    int x = startColIndex + (int)threadIdx.x;
    int y = startRowIndex + (int)blockIdx.x;
    int randIndex = wang_hash(x * y) % scene->width;

    curandState state;
    curand_init (x*y*(sampleIndex+2), 0, 0, &state);

    CudaThreadData threadData;
    threadData.debug = false;
    threadData.randState = state;
    threadData.randIndex = randIndex;
    threadData.scene = scene;

    int maxBounces = 8;
    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, scene->width, scene->height) - eye;
    Ray initialRay(eye, ray);

    float3 sampledColor = make_float3(0, 0, 0);
    float p = 0.0005f /* curand_normal(&threadData.randState)*/;
    //int numSamples = 8; //i.e. 8 * 4 = 32 or 1 * 4 = 4
    //for(int i=0; i<numSamples; i++) {
    initialRay.direction = initialRay.direction + make_float3(p, p, 0);
    sampledColor = sampledColor + traceSingleRay(initialRay, maxBounces, threadData);
    initialRay.direction = initialRay.direction + make_float3(-p, -p, 0);
    sampledColor = sampledColor + traceSingleRay(initialRay, maxBounces, threadData);
    initialRay.direction = initialRay.direction + make_float3(-p, p, 0);
    sampledColor = sampledColor + traceSingleRay(initialRay, maxBounces, threadData);
    initialRay.direction = initialRay.direction + make_float3(p, -p, 0);
    sampledColor = sampledColor + traceSingleRay(initialRay, maxBounces, threadData);
    //}
    //uchar4 lighting = toRGBA(sampledColor/((float)numSamples * 4.0f));
    uchar4 color = toRGBA(sampledColor/4.0f);

//    float3 cVal = traceSingleRay(eye, ray, maxBounces, threadData);
//    uchar4 lighting = toRGBA(cVal);
    if (sampleIndex != 0) {
        uchar4 ec;
        surf2Dread(&ec, image, x * sizeof(uchar4), scene->height - y, cudaBoundaryModeClamp);
        color = make_uchar4((color.x + ec.x)/2.0f, (color.y + ec.y)/2.0f, (color.z + ec.z)/2.0f, 255);
    }
    surf2Dwrite(color, image, x * sizeof(uchar4), scene->height - y, cudaBoundaryModeClamp);
}

__global__ void kernel_traceSingleRay(cudaSurfaceObject_t image, int x, int y, CudaScene* scene)
{
    int randIndex = (x * y) % scene->width;
    int maxBounces = 2;

    CudaThreadData threadData;
    curandState state;
    curand_init (x*y, 0, 0, &state);
    threadData.debug = true;
    threadData.randState = state;
    threadData.randIndex = randIndex;
    threadData.scene = scene;

    float3 eye = make_float3(0.0, 0.0, 0.0);
    float3 ray = cast_ray(x, y, scene->width, scene->height) - eye;
    Ray initialRay(eye, ray);
    printf("\n\nRay (%f, %f, %f)\n", ray.x, ray.y, ray.z);
    uchar4 color = toRGBA(traceSingleRay(initialRay, maxBounces, threadData));
    printf("Final Color: (%d, %d, %d, %d)\n", color.x, color.y, color.z, color.w);
    printf("Overall NumTriangleHits(%lu) NumMeshHitChecks(%lu)\n", numOverallTriangleChecks, numMeshHitChecks);
    printf("Overall average triangle hits: (%f)\n", (float)numOverallTriangleChecks/(float)numMeshHitChecks);

    uchar4 ec;
    surf2Dread(&ec, image, x * sizeof(uchar4), scene->height - y, cudaBoundaryModeClamp);
    printf("\nExisting lighting (%d, %d, %d, %d)\n", ec.x, ec.y, ec.z, ec.w);
    printf("\nNew lighting (%d, %d, %d, %d)\n", color.x, color.y, color.z, color.w);
    color = make_uchar4((color.x + ec.x)/2.0f, (color.y + ec.y)/2.0f, (color.z + ec.z)/2.0f, 255);
    printf("\nCombined lighting (%d, %d, %d, %d)\n", color.x, color.y, color.z, color.w);

    surf2Dwrite(color, image, x * sizeof(uchar4), scene->height - y, cudaBoundaryModeClamp);
}

__device__ uchar4 getColorAt(cudaSurfaceObject_t image, int2 imageDim, int2 index) {
    if (index.x < 0 || index.y < 0 || index.x >= imageDim.x || index.y >= imageDim.y) {
        return make_uchar4(255, 255, 255, 255);
    }
    uchar4 color;
    surf2Dread(&color, image, index.x * sizeof(uchar4), index.y, cudaBoundaryModeClamp);
    return color;
}

__device__ uchar4* getPixelColors(cudaSurfaceObject_t image, int2 dims, int2 imageIndex, float* kernel, int kDim, int count) {
    if (kDim % 2 == 0) return nullptr;

    int half = kDim / 2;
    auto colors = new uchar4[count != 0 ? count : kDim * kDim];
    int colIndex = 0;
    for(int i=0; i < kDim; i++) {
        for(int j=0; j < kDim; j++) {
            int index = i * kDim + j;
            if (kernel[index] != 0.0) {
                int xIndex = (imageIndex.x - half + j) % dims.x;
                int yIndex = (imageIndex.y - half + i) % dims.y;
                colors[colIndex++] = getColorAt(image, dims, make_int2(xIndex, yIndex));
            }
        }
    }
    return colors;
}

__global__ void kernel_denoise(cudaSurfaceObject_t image, int width, int height, int startRowIndex, int startColIndex, int sampleIndex) {
    int x = startColIndex + (int)threadIdx.x;
    int y = startRowIndex + (int)blockIdx.x;
    //uchar4 colors[5];
    //if (x > 0 && x < width-1 && y > 0 && y < height-1) {
    curandState state;
    curand_init (x*y*sampleIndex, 0, 0, &state);
    int numPixelsToSample = 5;
    float kernel[] = {0.0f, curand_normal(&state), 0.0f,
                      curand_normal(&state), curand_normal(&state), curand_normal(&state),
                      0.0f, curand_normal(&state), 0.0f };

//    float kernel[] = {0.0f, 0.0f, curand_normal(&state), 0.0f, 0.0f,
//                      0.0f, 0.0f, curand_normal(&state), 0.0f, 0.0f,
//                      curand_normal(&state), curand_normal(&state), curand_normal(&state), curand_normal(&state), curand_normal(&state),
//                      0.0f, 0.0f, curand_normal(&state), 0.0f, 0.0f,
//                      0.0f, 0.0f, curand_normal(&state), 0.0f, 0.0f};
    uchar4* colors = getPixelColors(image, make_int2(width, height), make_int2(x, y), kernel, 3, numPixelsToSample);

    if (colors != nullptr) {
        uchar4 color = colors[0];
        for (int i=1; i<numPixelsToSample; i++) {
            color.x = (color.x + colors[i].x)/2.0f;
            color.y = (color.y + colors[i].y)/2.0f;
            color.z = (color.z + colors[i].z)/2.0f;
        }
        surf2Dwrite(color, image, x * sizeof(uchar4), y, cudaBoundaryModeClamp);
        free(colors);
    }
}

//----------------------------------------------------------------------------------------------------------------------
//---------------------------------------------Cuda Utils Class Definition----------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

CudaKernelUtils::CudaKernelUtils() {}
CudaKernelUtils::~CudaKernelUtils() {}

void CudaKernelUtils::initializeRenderSurface(Texture* texture) {
    size_t stackLimit = 4096;
    cudaDeviceSetLimit(cudaLimitStackSize, stackLimit);
    size_t newHeapLimit = 33554432;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeapLimit);

    struct cudaGraphicsResource *vbo_res;
    // register this texture with CUDA
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

void CudaKernelUtils::renderScene(CudaScene* cudaScene, int blockSize, int numThreads, int startRowIndex, int startColIndex, int sampleIndex) {
    kernel_traceRays<<<blockSize, numThreads>>>(CudaKernelUtils::viewCudaSurfaceObject, cudaScene, startRowIndex, startColIndex, sampleIndex);
    check(cudaDeviceSynchronize());
    //test hits
//    Bounds* test = new Bounds(0.5, -0.5, -0.5, 0.5, -2.0, -3.0);
//    std::cout << "AABB HIT: " << checkHitOnAABB(make_float3(0.0, 0.0, 0.0), make_float3(0.0, 0.0, -1.0), test) << std::endl;
}

void CudaKernelUtils::runDenoiseKernel(CudaScene* cudaScene, int blockSize, int numThreads, int startRowIndex, int startColIndex, int sampleIndex) {
    kernel_denoise<<<blockSize, numThreads>>>(CudaKernelUtils::viewCudaSurfaceObject, cudaScene->width, cudaScene->height, startRowIndex, startColIndex, sampleIndex);
    check(cudaDeviceSynchronize());
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


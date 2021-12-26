#include "headers/cuda_allocation_utils.h"

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, const std::string& file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file.c_str(), line);
        exit(code);
    }
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
    T* hostPointer = (T*)malloc(len * sizeof(T));
    check(cudaMemcpy(hostPointer, src, len * sizeof(T), cudaMemcpyDeviceToHost))
    return hostPointer;
}

//-----------------------------------------------------------------------------------------------------------------------------
BVHBinaryNode* allocateBVH(BVHBinaryNode* node) {
    if (node == nullptr) {
        return node;
    }

    BVHBinaryNode tempNode = BVHBinaryNode();
    tempNode.left = allocateBVH(node->left);
    tempNode.right = allocateBVH(node->right);
    tempNode.bounds = cudaWrite<Bounds>(node->bounds, 1);
    if (node->numObjects != 0) {
        tempNode.objectsIndex = cudaWrite<int>(node->objectsIndex, node->numObjects);
        tempNode.numObjects = node->numObjects;
    }


    return cudaWrite<BVHBinaryNode>(&tempNode, 1);
}

CudaRTObject* allocateCudaObject(CudaRTObject* object) {
    CudaMaterial* matPtr = nullptr;
    if (object->material != nullptr) {
        matPtr = cudaWrite<CudaMaterial>(object->material, 1);
    }

    switch (object->type) {
        case SPHERE: {
            auto sphere = (CudaSphere*)object;
            CudaSphere tempSphere(sphere->position, sphere->radius, matPtr);
            return cudaWrite<CudaSphere>(&tempSphere, 1);
        }
        case MESH: {
            auto mesh = (CudaMesh*)object;
            auto cudaTrianglePtr = cudaWrite<CudaTriangle>(mesh->triangles, mesh->numTriangles);
            CudaMesh tempMesh(cudaTrianglePtr);
            tempMesh.numTriangles = mesh->numTriangles;
            tempMesh.material = matPtr;
            tempMesh.bvhRoot = allocateBVH(mesh->bvhRoot);

            return cudaWrite<CudaMesh>(&tempMesh, 1);
        }
    }
    return nullptr;
}

CudaRTObject* allocateCudaLight(CudaRTObject* light) {
    switch (((CudaLight*)light)->lightType) {
        case SKYBOX_LIGHT: {
            auto skyboxLight = (CudaSkyboxLight*)light;
            CudaSkyboxLight tempLight((CudaSphere*)allocateCudaObject(skyboxLight->sphere));
            return cudaWrite<CudaSkyboxLight>(&tempLight, 1);
        }
        case POINT_LIGHT: {
            return nullptr;
        }
    }
    return nullptr;
}

CudaRTObject** allocateCudaObjects(CudaScene* scene) {
    int numObjects = scene->numObjects;
    auto objects = new CudaRTObject*[numObjects];
    for (int i=0; i<numObjects; i++) {
        CudaRTObject* cudaPtr = allocateCudaObject(scene->objects[i]);
        objects[i] = cudaPtr;
    }
    auto cudaObjectsPtr = cudaWrite<CudaRTObject *>(objects, numObjects);
    delete[] objects;

    return cudaObjectsPtr;
}

CudaRTObject** allocateCudaLights(CudaScene* scene) {
    int numLights = scene->numLights;
    auto lights = new CudaRTObject*[numLights];
    for (int i=0; i < numLights; i++) {
        lights[i] = allocateCudaLight(scene->hostLights->at(i));
    }
    auto cudaLightsPtr = cudaWrite<CudaRTObject*>(lights, numLights);
    delete[] lights;

    return cudaLightsPtr;
}

CudaRandomGenerator* allocateRandomGenerator(int num) {
    CudaRandomGenerator tempGenerator;
    auto numbers = new float[num];
    for(int i=0; i<num; i++) {
        numbers[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    tempGenerator.randomNumbers = cudaWrite<float>(numbers, num);
    tempGenerator.numRand = num;
    tempGenerator.index = 0;
    delete[] numbers;
    return cudaWrite<CudaRandomGenerator>(&tempGenerator, 1);
}

CudaScene* allocateCudaScene(CudaScene* scene) {
    CudaRTObject** cudaObjectsPtr = allocateCudaObjects(scene);
    CudaRTObject** cudaLightsPtr = allocateCudaLights(scene);
    CudaScene tempScene(cudaObjectsPtr, scene->numObjects, cudaLightsPtr, scene->numLights);
    tempScene.width = scene->width;
    tempScene.height = scene->height;
    tempScene.generator = allocateRandomGenerator(scene->width);
    return cudaWrite<CudaScene>(&tempScene, 1);
}

void cleanCudaScene(CudaScene* scene) {
    for (int i=0; i<scene->numObjects; i++) {
        cudaFree(scene->objects[i]);
    }
    cudaFree(scene);
}
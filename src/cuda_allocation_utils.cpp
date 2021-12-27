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
//-----------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------

BVHBinaryNode* allocateBVH(BVHBinaryNode* node) {
    if (node == nullptr) {
        return node;
    }

    BVHBinaryNode tempNode = BVHBinaryNode();
    tempNode.left = allocateBVH(node->left);
    tempNode.right = allocateBVH(node->right);
    if (node->bounds != nullptr) {
        tempNode.bounds = cudaWrite<Bounds>(node->bounds, 1);
    }
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

CudaScene* allocateCudaScene(CudaScene* scene) {
    CudaRTObject** cudaObjectsPtr = allocateCudaObjects(scene);
    CudaRTObject** cudaLightsPtr = allocateCudaLights(scene);
    CudaScene tempScene(cudaObjectsPtr, scene->numObjects, cudaLightsPtr, scene->numLights);
    tempScene.width = scene->width;
    tempScene.height = scene->height;
    return cudaWrite<CudaScene>(&tempScene, 1);
}

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

void freeBVH(BVHBinaryNode* allocatedNode) {
    if (allocatedNode == nullptr) return;

    auto node = cudaRead<BVHBinaryNode>(allocatedNode, 1);

    freeBVH(node->left);
    freeBVH(node->right);

    if (node->bounds != nullptr) cudaFree(node->bounds);
    if (node->numObjects != 0) cudaFree(node->objectsIndex);

    delete node;
    cudaFree(allocatedNode);
}

void freeCudaObject(CudaRTObject* allocatedObject) {
    auto object = cudaRead<CudaRTObject>(allocatedObject, 1);
    if (object->material != nullptr) {
        cudaFree(object->material);
    }
    switch (object->type) {
        case SPHERE: {
            cudaFree((CudaSphere*)allocatedObject);
            break;
        }
        case MESH: {
            auto mesh = cudaRead<CudaMesh>((CudaMesh*)allocatedObject, 1);
            cudaFree(mesh->triangles);
            freeBVH(mesh->bvhRoot);

            delete mesh;
            cudaFree((CudaMesh*)allocatedObject);
            break;
        }
    }

    delete object;
}

void freeCudaLight(CudaRTObject* allocatedLight) {
    auto light = cudaRead<CudaLight>((CudaLight*)allocatedLight, 1);
    switch (light->lightType) {
        case SKYBOX_LIGHT: {
            auto skyboxLight = cudaRead<CudaSkyboxLight>((CudaSkyboxLight*)allocatedLight, 1);
            freeCudaObject(skyboxLight->sphere);
            delete skyboxLight;
            cudaFree((CudaSkyboxLight*)allocatedLight);
            break;
        }
        case POINT_LIGHT: {

            break;
        }
    }
    delete light;
}

void cleanCudaScene(CudaScene* allocatedScene) {
    auto scene = cudaRead<CudaScene>(allocatedScene, 1);
    scene->objects = cudaRead<CudaRTObject*>(scene->objects, scene->numObjects);
    scene->lights = cudaRead<CudaRTObject*>(scene->lights, scene->numLights);
    for (int i=0; i<scene->numObjects; i++) {
        freeCudaObject(scene->objects[i]);
    }

    for (int i=0; i<scene->numLights; i++) {
        freeCudaLight(scene->lights[i]);
    }

    delete scene->objects;
    delete scene->lights;
    delete scene;

    cudaFree(allocatedScene);
}
#include "headers/cuda_allocation_utils.h"

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
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

CudaRTObject* allocateCudaObjects(CudaRTObject* object) {
    switch (object->type) {
        case SPHERE: {
            auto sphere = (CudaSphere*)object;
            CudaSphere tempSphere(sphere->position, sphere->radius, cudaWrite<CudaMaterial>(sphere->material, 1));
            return cudaWrite<CudaSphere>(&tempSphere, 1);
        }
        case MESH: {
            auto mesh = (CudaMesh*)object;
            auto cudaTrianglePtr = cudaWrite<CudaTriangle>(mesh->triangles, mesh->numTriangles);
            CudaMesh tempMesh(cudaTrianglePtr);
            tempMesh.numTriangles = mesh->numTriangles;
            tempMesh.material = cudaWrite<CudaMaterial>(mesh->material, 1);
            tempMesh.bvhRoot = allocateBVH(mesh->bvhRoot);

            return cudaWrite<CudaMesh>(&tempMesh, 1);
        }
    }
    return nullptr;
}

CudaScene* allocateCudaScene(CudaScene* scene) {
    int numObjects = scene->numObjects;
    auto objects = new CudaRTObject*[numObjects];
    for (int i=0; i<numObjects; i++) {
        CudaRTObject* cudaPtr = allocateCudaObjects(scene->objects[i]);
        objects[i] = cudaPtr;
    }
    CudaRTObject** cudaObjectsPtr = cudaWrite<CudaRTObject *>(objects, numObjects);
    delete[] objects;

    CudaScene tempScene(cudaObjectsPtr, numObjects);
    return cudaWrite<CudaScene>(&tempScene, 1);
}

void cleanCudaScene(CudaScene* scene) {
    for (int i=0; i<scene->numObjects; i++) {
        cudaFree(scene->objects[i]);
    }
    cudaFree(scene);
}
#include "headers/cuda_helper_utils.h"

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
    delete node->left->bounds;
    delete node->right->bounds;
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

BVHBinaryNode* createHostTreeHelper(std::vector<CudaTriangle*>* localTriangles, BVHBinaryNode* node) {
    int len = localTriangles->size();
    if (len <= 5) {
        int* indices = new int[len];
        for (int i=0; i<len; i++) {
            indices[i] = localTriangles->at(i)->index;
        }
        node->objectsIndex = indices;
        return node;
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
    delete node->left->bounds;
    delete node->right->bounds;
    node->left->bounds = getNewBounds(leftTriangles);
    node->right->bounds = getNewBounds(rightTriangles);

    node->left = createTreeHelper(leftTriangles, node->left);
    delete leftTriangles;
    node->right = createTreeHelper(rightTriangles, node->right);
    delete rightTriangles;

    node->objectsIndex = currNodeIndices.data();
    node->numObjects = currNodeIndices.size();
    return node;
}

//-----------------------------------------------------------------------------------------------------------------------------

CudaRTObject* allocateCudaObjects(CudaRTObject* object) {
    switch (object->type) {
        case SPHERE: {
            auto sphere = (CudaSphere*)object;
            CudaSphere tempSphere(sphere->position, sphere->radius, cudaWrite<CudaMaterial>(sphere->material, 1));
            return cudaWrite<CudaSphere>(&tempSphere, 1);
        }
        case MESH: {
            auto mesh = (CudaMesh*)object;
            auto treeTriangles = new std::vector<CudaTriangle*>();
            for(int i=0; i<mesh->numTriangles; i++) {
                CudaTriangle t = mesh->triangles[i];
                treeTriangles->push_back(&t);
            }
            auto cudaTrianglePtr = cudaWrite<CudaTriangle>(mesh->triangles, mesh->numTriangles);
            CudaMesh tempMesh(cudaTrianglePtr);
            tempMesh.numTriangles = mesh->numTriangles;
            tempMesh.material = cudaWrite<CudaMaterial>(mesh->material, 1);

            BVHBinaryNode root(getNewBounds(treeTriangles));
            tempMesh.bvhRoot = createTreeHelper(treeTriangles, &root);

            delete treeTriangles;
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
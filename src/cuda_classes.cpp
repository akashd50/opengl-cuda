#include "headers/cuda_classes.h"

float3 vec3ToFloat3(glm::vec3 vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

Bounds* getNewBounds(std::vector<CudaTriangle>* triangles, std::vector<int>* indices) {
    auto b = new Bounds();
    for(int index : *indices) {
        CudaTriangle t = triangles->at(index);
        float maxX = std::fmax(std::fmax(t.a.x, t.b.x), t.c.x);
        float minX = std::fmin(std::fmin(t.a.x, t.b.x), t.c.x);
        float maxY = std::fmax(std::fmax(t.a.y, t.b.y), t.c.y);
        float minY = std::fmin(std::fmin(t.a.y, t.b.y), t.c.y);
        float maxZ = std::fmax(std::fmax(t.a.z, t.b.z), t.c.z);
        float minZ = std::fmin(std::fmin(t.a.z, t.b.z), t.c.z);
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

BVHBinaryNode* createMeshTree(std::vector<CudaTriangle>* triangles, std::vector<int>* indices, BVHBinaryNode* node) {
    int len = indices->size();
    if (len <= 5) {
        int* localIndices = new int[len];
        for (int i=0; i<len; i++) { localIndices[i] = indices->at(i); }
        node->objectsIndex = localIndices;
        return node;
    }

    auto leftTriangles = new std::vector<int>();
    auto rightTriangles = new std::vector<int>();

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
    for (int index : *indices) {
        //divide along the axis with max length
        CudaTriangle t = triangles->at(index);
        if (isTriangleInBounds(&t, node->left->bounds)) {
            leftTriangles->push_back(index);
        }
        else if (isTriangleInBounds(&t, node->right->bounds)) {
            rightTriangles->push_back(index);
        } else {
            currNodeIndices.push_back(index);
        }
    }

    delete node->left->bounds;
    delete node->right->bounds;
    node->left->bounds = getNewBounds(triangles, leftTriangles);
    node->right->bounds = getNewBounds(triangles, rightTriangles);

    node->left = createMeshTree(triangles, leftTriangles, node->left);
    delete leftTriangles;
    node->right = createMeshTree(triangles, rightTriangles, node->right);
    delete rightTriangles;

    node->objectsIndex = currNodeIndices.data();
    node->numObjects = currNodeIndices.size();
    return node;
}
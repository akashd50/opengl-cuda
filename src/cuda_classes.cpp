#include "headers/cuda_classes.h"

//----------------------------------------------------------------------------------------------------------------------

float3 vec3ToFloat3(glm::vec3 vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

Bounds* updateBounds(std::vector<CudaTriangle>* triangles, std::vector<int>* indices, Bounds* b) {
    b->reset();
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

float3 getTrianglePosition(float3 a, float3 b, float3 c) {
    return make_float3((a.x + b.x + c.x)/3.0f, (a.y + b.y + c.y)/3.0f, (a.z + b.z + c.z)/3.0f);
}

bool isTriangleCentroidInBounds(CudaTriangle* triangle, Bounds* bounds) {
    float3 pos = getTrianglePosition(triangle->a, triangle->b, triangle->c);
    return (pos.x >= bounds->left && pos.x <= bounds->right) &&
           (pos.y >= bounds->bottom && pos.y <= bounds->top) &&
           (pos.z >= bounds->back && pos.z <= bounds->front);
}

//----------------------------------------------------------------------------------------------------------------------

CudaMaterial::CudaMaterial(float3 _ambient, float3 _diffuse) : ambient(_ambient), diffuse(_diffuse),
                            specular(make_float3(0, 0, 0)), shininess(1),
                            reflective(make_float3(0, 0, 0)), transmissive(make_float3(0, 0, 0)),
                            refraction(0) {}

CudaMaterial::CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float _shininess) :
        ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
        reflective(make_float3(0, 0, 0)), transmissive(make_float3(0, 0, 0)), refraction(0) {}

CudaMaterial::CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float3 _reflective,
                           float3 _transmissive, float _refraction, float _roughness, float _shininess) :
        ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
        reflective(_reflective), transmissive(_transmissive), refraction(_refraction), roughness(_roughness) {}

//----------------------------------------------------------------------------------------------------------------------

int CudaRTObject::OBJECT_ID = 0;

CudaRTObject::CudaRTObject()
: id(CudaRTObject::OBJECT_ID++), material(nullptr) {}

CudaRTObject::CudaRTObject(int _type)
: type(_type), id(CudaRTObject::OBJECT_ID++), material(nullptr) {}

CudaRTObject::CudaRTObject(int _type, CudaMaterial* _material)
: type(_type), material(_material), id(CudaRTObject::OBJECT_ID++) {}

CudaRTObject::CudaRTObject(int _type, CudaMaterial* _material, float3 _position)
: type(_type), material(_material), position(_position), id(CudaRTObject::OBJECT_ID++) {}

//----------------------------------------------------------------------------------------------------------------------

CudaTriangle::CudaTriangle(float3 _a, float3 _b, float3 _c): a(_a), b(_b), c(_c) {}
CudaTriangle::CudaTriangle(float3 _a, float3 _b, float3 _c, int _index): a(_a), b(_b), c(_c), index(_index) {}

//----------------------------------------------------------------------------------------------------------------------

CudaSphere::CudaSphere(float3 _position, float _radius):
CudaRTObject(SPHERE, nullptr, _position), radius(_radius) {}

CudaSphere::CudaSphere(float3 _position, float _radius, CudaMaterial* _material):
CudaRTObject(SPHERE, _material, _position), radius(_radius) {}

//----------------------------------------------------------------------------------------------------------------------

CudaMesh::CudaMesh(): CudaRTObject(MESH) {}

CudaMesh::CudaMesh(CudaTriangle* _triangles): CudaRTObject(MESH), triangles(_triangles) {}

void CudaMesh::addTriangle(CudaTriangle _object) {
    hostTriangles->push_back(_object);
    triangles = hostTriangles->data();
    numTriangles++;
}

void CudaMesh::finalize() {
    auto allIndices = new std::vector<int>();
    for (int i=0; i<hostTriangles->size(); i++) allIndices->push_back(i);
    maxBVHDepth = 0;
    bvhRoot = createMeshTree2(hostTriangles, allIndices, bvhRoot, 0);
    dimensions = make_float3(bvhRoot->bounds->right - bvhRoot->bounds->left,
                             bvhRoot->bounds->top - bvhRoot->bounds->bottom,
                             bvhRoot->bounds->front - bvhRoot->bounds->back);
}

CudaMesh* CudaMesh::newHostMesh() {
    auto mesh = new CudaMesh();
    mesh->hostTriangles = new std::vector<CudaTriangle>();
    //mesh->triangles = mesh->hostTriangles->data();
    mesh->numTriangles = 0;
    mesh->bvhRoot = new BVHBinaryNode();
    return mesh;
}

BVHBinaryNode* CudaMesh::createMeshTree(std::vector<CudaTriangle>* localTriangles, std::vector<int>* indices, BVHBinaryNode* node, int depth) {
    maxBVHDepth = std::max(depth, maxBVHDepth);
    int len = indices->size();
    if (len <= 5) {
        int* localIndices = new int[len];
        for (int i=0; i<len; i++) { localIndices[i] = indices->at(i); }
        node->objectsIndex = localIndices;
        node->numObjects = len;
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

    auto currNodeIndices = new std::vector<int>();
    for (int index : *indices) {
        //divide along the axis with max length
        CudaTriangle t = localTriangles->at(index);
        if (isTriangleInBounds(&t, node->left->bounds)) {
            leftTriangles->push_back(index);
        }
        else if (isTriangleInBounds(&t, node->right->bounds)) {
            rightTriangles->push_back(index);
        } else {
            currNodeIndices->push_back(index);
        }
    }

    node->left->bounds = updateBounds(localTriangles, leftTriangles, node->left->bounds);
    node->right->bounds = updateBounds(localTriangles, rightTriangles, node->right->bounds);

    node->left = createMeshTree(localTriangles, leftTriangles, node->left, depth + 1);
    delete leftTriangles;
    node->right = createMeshTree(localTriangles, rightTriangles, node->right, depth + 1);
    delete rightTriangles;

    node->objectsIndex = currNodeIndices->data();
    node->numObjects = currNodeIndices->size();
    return node;
}

BVHBinaryNode* CudaMesh::createMeshTree2(std::vector<CudaTriangle>* localTriangles, std::vector<int>* indices, BVHBinaryNode* node, int depth) {
    maxBVHDepth = std::max(depth, maxBVHDepth);
    int len = indices->size();
    if (len <= 20 || depth >= 24) {
        int* localIndices = new int[len];
        for (int i=0; i<len; i++) { localIndices[i] = indices->at(i); }
        node->objectsIndex = localIndices;
        node->numObjects = len;
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
    else {
        //if (zLen >= yLen && zLen >= xLen) {
        //zDiv = true;
        float mid = (nb.front + nb.back)/2;
        node->left = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, nb.right, mid, nb.back));
        node->right = new BVHBinaryNode(new Bounds(nb.top, nb.bottom, nb.left, nb.right, nb.front, mid));
    }

    for (int index : *indices) {
        //divide along the axis with max length
        CudaTriangle t = localTriangles->at(index);
        if (isTriangleCentroidInBounds(&t, node->left->bounds)) {
            leftTriangles->push_back(index);
        }
        else if (isTriangleCentroidInBounds(&t, node->right->bounds)) {
            rightTriangles->push_back(index);
        }
    }

    node->left->bounds = updateBounds(localTriangles, leftTriangles, node->left->bounds);
    node->right->bounds = updateBounds(localTriangles, rightTriangles, node->right->bounds);

    node->left = createMeshTree2(localTriangles, leftTriangles, node->left, depth + 1);
    delete leftTriangles;
    node->right = createMeshTree2(localTriangles, rightTriangles, node->right, depth + 1);
    delete rightTriangles;

    return node;
}

//----------------------------------------------------------------------------------------------------------------------

CudaLight::CudaLight(): CudaRTObject(), intensity(1.0f) {}

CudaLight::CudaLight(int _lightType): CudaRTObject(LIGHT), lightType(_lightType), intensity(1.0f) {}

CudaLight::CudaLight(int _lightType, float3 _color): CudaRTObject(LIGHT), lightType(_lightType), intensity(1.0f) {}

//----------------------------------------------------------------------------------------------------------------------

CudaSkyboxLight::CudaSkyboxLight(): CudaLight(SKYBOX_LIGHT) {}

CudaSkyboxLight::CudaSkyboxLight(CudaSphere* _sphere): CudaLight(SKYBOX_LIGHT), sphere(_sphere) {}

//----------------------------------------------------------------------------------------------------------------------

CudaPointLight::CudaPointLight(float3 _position)
: CudaLight(POINT_LIGHT), position(_position) {}

CudaPointLight::CudaPointLight(float3 _position, float3 _color)
: CudaLight(POINT_LIGHT), position(_position) {}

//----------------------------------------------------------------------------------------------------------------------

CudaMeshLight::CudaMeshLight()
: CudaLight(MESH_LIGHT), mesh(nullptr) {}

CudaMeshLight::CudaMeshLight(CudaMesh* _mesh)
: CudaLight(MESH_LIGHT), mesh(_mesh) {}

CudaMeshLight::CudaMeshLight(CudaMesh* _mesh, float3 _color)
: CudaLight(MESH_LIGHT), mesh(_mesh), color(_color) {}

//----------------------------------------------------------------------------------------------------------------------

CudaScene::CudaScene(): numObjects(0) {};
CudaScene::CudaScene(CudaRTObject** _objects , int _numObjects): objects(_objects), numObjects(_numObjects) {}
CudaScene::CudaScene(CudaRTObject** _objects , int _numObjects, CudaRTObject** _lights , int _numLights)
: objects(_objects), numObjects(_numObjects), lights(_lights), numLights(_numLights) {};

void CudaScene::addObject(CudaRTObject* _object) {
    hostObjects->push_back(_object);
    objects = hostObjects->data();
    numObjects = hostObjects->size();
}

void CudaScene::addLight(CudaRTObject* _light) {
    hostLights->push_back(_light);
    lights = hostLights->data();
    numLights = hostLights->size();
}

CudaScene* CudaScene::newHostScene() {
    auto scene = new CudaScene();
    scene->hostObjects = new std::vector<CudaRTObject*>();
    scene->objects = scene->hostObjects->data();

    scene->hostLights = new std::vector<CudaRTObject*>();
    scene->lights = scene->hostLights->data();
    scene->numObjects = 0;
    scene->numLights = 0;
    return scene;
}

//----------------------------------------------------------------------------------------------------------------------

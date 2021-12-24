#pragma once
#include <glm/glm.hpp>
#include <vector>
#include "bvh_classes.h"
#include "ShaderConst.h"

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------CUDA--OBJECTS-----------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
float3 vec3ToFloat3(glm::vec3 vec);

class CudaMaterial {
public:
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float shininess;

    float3 reflective;
    float3 transmissive;
    float refraction, roughness;

    CudaMaterial() :
            ambient(make_float3(0, 0, 0)), diffuse(make_float3(0, 0, 0)), specular(make_float3(0, 0, 0)), shininess(1),
            reflective(make_float3(0, 0, 0)), transmissive(make_float3(0, 0, 0)), refraction(0) {}
    CudaMaterial(float3 _ambient, float3 _diffuse) :
            ambient(_ambient), diffuse(_diffuse), specular(make_float3(0, 0, 0)), shininess(1),
            reflective(make_float3(0, 0, 0)), transmissive(make_float3(0, 0, 0)), refraction(0) {}
    CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float _shininess) :
            ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
            reflective(make_float3(0, 0, 0)), transmissive(make_float3(0, 0, 0)), refraction(0) {}
    CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float _shininess,
                 float3 _reflective, float3 _transmissive, float _refraction) :
            ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
            reflective(_reflective), transmissive(_transmissive), refraction(_refraction) {}
    CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float3 _reflective,
                 float3 _transmissive, float _refraction, float _roughness, float _shininess) :
            ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
            reflective(_reflective), transmissive(_transmissive), refraction(_refraction), roughness(_roughness) {}
};

class CudaRTObject {
public:
    int type;
    CudaMaterial* material;
    inline CudaRTObject(int _type) : type(_type) {}
    inline CudaRTObject(int _type, CudaMaterial* _material) : type(_type), material(_material) {}
};

class CudaTriangle {
public:
    float3 a, b, c;
    int index;
    //CudaTriangle(float3 _a, float3 _b, float3 _c): CudaRTObject(TRIANGLE), a(_a), b(_b), c(_c) {}
    CudaTriangle(float3 _a, float3 _b, float3 _c): a(_a), b(_b), c(_c) {}
    CudaTriangle(float3 _a, float3 _b, float3 _c, int _index): a(_a), b(_b), c(_c), index(_index) {}

    float3 getPosition() {
        return make_float3((a.x + b.x + c.x)/3, (a.y + b.y + c.y)/3, (a.z + b.z + c.z)/3);
    }
};

class CudaMesh: public CudaRTObject {
public:
    std::vector<CudaTriangle>* hostTriangles;
    CudaTriangle* triangles;
    int numTriangles;
    BVHBinaryNode* bvhRoot;
    int maxBVHDepth;
    CudaMesh(): CudaRTObject(MESH) {}
    CudaMesh(CudaTriangle* _triangles): CudaRTObject(MESH), triangles(_triangles) {}

    void addTriangle(CudaTriangle _object) {
        hostTriangles->push_back(_object);
        triangles = hostTriangles->data();
        numTriangles = hostTriangles->size();
    }

    void finalize() {
        auto allIndices = new std::vector<int>();
        for (int i=0; i<hostTriangles->size(); i++) allIndices->push_back(i);
        bvhRoot = createMeshTree(hostTriangles, allIndices, bvhRoot);
    }

    BVHBinaryNode* createMeshTree(std::vector<CudaTriangle>* triangles, std::vector<int>* indices, BVHBinaryNode* node);

    static CudaMesh* newHostMesh() {
        auto mesh = new CudaMesh();
        mesh->hostTriangles = new std::vector<CudaTriangle>();
        mesh->triangles = mesh->hostTriangles->data();
        mesh->numTriangles = 0;
        mesh->bvhRoot = new BVHBinaryNode();
        return mesh;
    }
};

class CudaSphere: public CudaRTObject {
public:
    float3 position;
    float radius;
    CudaSphere(float3 _position, float _radius, CudaMaterial* _material): CudaRTObject(SPHERE, _material),
                                                                                 position(_position), radius(_radius) {}
};

class CudaScene {
public:
    std::vector<CudaRTObject*>* hostObjects;
    CudaRTObject** objects;
    int numObjects;
    CudaScene(): numObjects(0) {};
    CudaScene(CudaRTObject** _objects , int _numObjects): objects(_objects), numObjects(_numObjects) {}

    void addObject(CudaRTObject* _object) {
        hostObjects->push_back(_object);
        objects = hostObjects->data();
        numObjects = hostObjects->size();
    }

    static CudaScene* newHostScene() {
        auto scene = new CudaScene();
        scene->hostObjects = new std::vector<CudaRTObject*>();
        scene->objects = scene->hostObjects->data();
        return scene;
    }
};

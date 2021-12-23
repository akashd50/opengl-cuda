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
    CudaTriangle* triangles;
    int numTriangles;
    BVHBinaryNode* bvhRoot;
    int maxBVHDepth;
    CudaMesh(CudaTriangle* _triangles): CudaRTObject(MESH), triangles(_triangles) {}
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
    CudaRTObject** objects;
    int numObjects;
    CudaScene(CudaRTObject** _objects , int _numObjects): objects(_objects), numObjects(_numObjects) {}
};

//-------------------------

class Material {
public:
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    glm::vec3 reflective;
    glm::vec3 transmissive;
    float refraction, roughness, shininess;;

    Material() :
            ambient(glm::vec3(0, 0, 0)), diffuse(glm::vec3(0, 0, 0)), specular(glm::vec3(0, 0, 0)), shininess(1),
            reflective(glm::vec3(0, 0, 0)), transmissive(glm::vec3(0, 0, 0)), refraction(0) {}
    Material(glm::vec3 _ambient, glm::vec3 _diffuse) :
            ambient(_ambient), diffuse(_diffuse), specular(glm::vec3(0, 0, 0)), shininess(1),
            reflective(glm::vec3(0, 0, 0)), transmissive(glm::vec3(0, 0, 0)), refraction(0) {}
    Material(glm::vec3 _ambient, glm::vec3 _diffuse, glm::vec3 _specular, float _shininess) :
            ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
            reflective(glm::vec3(0, 0, 0)), transmissive(glm::vec3(0, 0, 0)), refraction(0) {}
    Material(glm::vec3 _ambient, glm::vec3 _diffuse, glm::vec3 _specular, float _shininess,
             glm::vec3 _reflective, glm::vec3 _transmissive, float _refraction) :
            ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
            reflective(_reflective), transmissive(_transmissive), refraction(_refraction) {}
    Material(glm::vec3 _ambient, glm::vec3 _diffuse, glm::vec3 _specular, float _shininess,
             glm::vec3 _reflective, glm::vec3 _transmissive, float _refraction, float _roughness) :
            ambient(_ambient), diffuse(_diffuse), specular(_specular), shininess(_shininess),
            reflective(_reflective), transmissive(_transmissive), refraction(_refraction), roughness(_roughness) {}

//    CudaMaterial toCudaMaterial() {
//        return {vec3ToFloat3(ambient), vec3ToFloat3(diffuse), vec3ToFloat3(specular),
//                            vec3ToFloat3(reflective), vec3ToFloat3(transmissive),
//                            refraction, roughness, shininess};
//    }

    CudaMaterial* toNewCudaMaterial() {
        return new CudaMaterial(vec3ToFloat3(ambient), vec3ToFloat3(diffuse), vec3ToFloat3(specular),
                vec3ToFloat3(reflective), vec3ToFloat3(transmissive),
                refraction, roughness, shininess);
    }
};

class RTObject {
public:
    int type;
    Material* material;
    RTObject( int _type) : type (_type) {}

    virtual CudaRTObject* toNewCudaObject() = 0;

    ~RTObject() {
        delete material;
    }
};

class Triangle {
public:
    glm::vec3 a, b, c;
    Triangle(glm::vec3 _a, glm::vec3 _b, glm::vec3 _c): a(_a), b(_b), c(_c) {}

    CudaTriangle toCudaObject() {
        return {vec3ToFloat3(a), vec3ToFloat3(b), vec3ToFloat3(c)};
    }

    CudaTriangle* toNewCudaObject() {
        return new CudaTriangle(vec3ToFloat3(a), vec3ToFloat3(b), vec3ToFloat3(c));
    }
};

class Mesh: public RTObject {
public:
    std::vector<Triangle*>* triangles;
    Bounds* bounds;
    Mesh(): RTObject(MESH) {
        //top = 0; bottom = 0; left = 0; right = 0; front = 0; back = 0;
        bounds = new Bounds(-9999, 9999, 9999, -9999, -9999, 9999);
        triangles = new std::vector<Triangle*>();
    }

    Mesh(std::vector<Triangle*>* _triangles): RTObject(MESH), triangles(_triangles) {}

    void addTriangle(Triangle* triangle) {
        triangles->push_back(triangle);
    }

//    CudaMesh toCudaMesh() {
//        auto tempTriangles = new std::vector<CudaTriangle>();
//        for(int i=0; i<triangles->size(); i++) {
//            CudaTriangle cudaTriangle = triangles->at(i)->toCudaTriangle();
//            cudaTriangle.index = i;
//            tempTriangles->push_back(cudaTriangle);
//        }
//        return {tempTriangles->data()};
//    }

    CudaMesh* toNewCudaObject() {
        auto tempTriangles = new std::vector<CudaTriangle>();
        for(int i=0; i<triangles->size(); i++) {
            CudaTriangle cudaTriangle = triangles->at(i)->toCudaObject();
            cudaTriangle.index = i;
            tempTriangles->push_back(cudaTriangle);
        }
        CudaMesh* mesh = new CudaMesh(tempTriangles->data());
        mesh->numTriangles = tempTriangles->size();
        mesh->material = material->toNewCudaMaterial();
        return mesh;
    }
};

class Sphere: public RTObject {
public:
    glm::vec3 position;
    float radius;

    Sphere() : RTObject(SPHERE) {}

    Sphere(Material* _material, float _radius, glm::vec3 _position): RTObject(SPHERE) {
        material = _material;
        Sphere::radius = _radius;
        Sphere::position = _position;
    }

    CudaSphere* toNewCudaObject() {
        return new CudaSphere(vec3ToFloat3(position), radius, material->toNewCudaMaterial());
    }
};

class Scene {
    std::vector<RTObject*>* objects;
public:
    Scene() {
        Scene::objects = new std::vector<RTObject*>();
    }

    void addObject(RTObject* _object) {
        Scene::objects->push_back(_object);
    }

    std::vector<RTObject*> getObjects() {
        return *objects;
    }

    CudaScene* toNewCudaScene() {
        int numObjects = objects->size();
        auto cudaObjects = new CudaRTObject*[numObjects];
        int index = 0;
        for (RTObject* obj : *objects) {
            CudaRTObject* newObject = obj->toNewCudaObject();
            cudaObjects[index++] = newObject;
        }
        return new CudaScene(cudaObjects, index);
    }

    ~Scene() {
        for(RTObject* object : *objects) {
            delete object;
        }
        delete objects;
    }
};
#pragma once
#include <vector_types.h>
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

__device__ const int SPHERE = 1;
__device__ const int MESH = 2;
__device__ const int TRIANGLE = 3;
__device__ const int PLANE = 4;

//----------------------------------------------------------------------------------------------------------------------

class Bounds {
public:
    float top, bottom, left, right, front, back;
    Bounds(): top(-9999), bottom(9999), left(9999), right(-9999), front(-9999), back(9999) {}
    Bounds(float _t, float _b, float _l, float _r, float _f, float _back): top(_t), bottom(_b),
                                                                           left(_l), right(_r), front(_f), back(_back) {}
};

//----------------------------------------------------------------------------------------------------------------------

//class BVHNode {
//public:
//    BVHNode *top1, *top2, *top3, *top4, *bottom1, *bottom2, *bottom3, *bottom4;
//    int objectIndex;
//    BVHNode() {
//
//    }
//};

class BVHBinaryNode {
public:
    BVHBinaryNode *left, *right;
    Bounds* bounds;
    int* objectsIndex;
    int numObjects;
    BVHBinaryNode(): numObjects(0) {}
    BVHBinaryNode(Bounds* _bounds): bounds(_bounds), left(nullptr), right(nullptr), objectsIndex(nullptr), numObjects(0) {}
    BVHBinaryNode(Bounds* _bounds, int* _objectsIndex, int _numObject): bounds(_bounds), objectsIndex(_objectsIndex),
                                                                        left(nullptr), right(nullptr), numObjects(_numObject) {}
    BVHBinaryNode(Bounds* _bounds, BVHBinaryNode* _left, BVHBinaryNode* _right): bounds(_bounds),
                                                                                 left(_left), right(_right), objectsIndex(nullptr), numObjects(0) {}
};

//----------------------------------------------------------------------------------------------------------------------

class Material {
public:
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;

    glm::vec3 reflective;
    glm::vec3 transmissive;
    float refraction, roughness;

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
};

class RTObject {
private:
    int type;
    Material* material;
public:
    RTObject( int _type) : type (_type) {}

    void setMaterial(Material* _material) {
        RTObject::material = _material;
    }

    Material* getMaterial() {
        return material;
    }

    int getType() {
        return type;
    }

    ~RTObject() {
        delete material;
    }
};

class Triangle {
public:
    glm::vec3 a, b, c;
    Triangle(glm::vec3 _a, glm::vec3 _b, glm::vec3 _c): a(_a), b(_b), c(_c) {}
};

class Mesh: public RTObject {
public:
    std::vector<Triangle*>* triangles;
    //float top, bottom, left, right, front, back;
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
};

class Sphere: public RTObject {
private:
    glm::vec3 position;
    float radius;

public:
    inline Sphere() : RTObject(SPHERE) {}

    inline Sphere(Material* _material, float _radius, glm::vec3 _position): RTObject(SPHERE) {
        RTObject::setMaterial(_material);
        Sphere::radius = _radius;
        Sphere::position = _position;
    }

    inline glm::vec3 getPosition() {
        return position;
    }

    inline void setPosition(const glm::vec3 &_position) {
        Sphere::position = _position;
    }

    inline float getRadius() const {
        return radius;
    }

    inline void setRadius(float _radius) {
        Sphere::radius = _radius;
    }
};

class Scene {
    std::vector<RTObject*>* objects;
public:
    inline Scene() {
        Scene::objects = new std::vector<RTObject*>();
    }

    inline void addObject(RTObject* _object) {
        Scene::objects->push_back(_object);
    }

    inline std::vector<RTObject*> getObjects() {
        return *objects;
    }

    inline ~Scene() {
        for(RTObject* object : *objects) {
            delete object;
        }
        delete objects;
    }
};

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------CUDA--OBJECTS-----------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

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
    CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float _shininess,
                 float3 _reflective, float3 _transmissive, float _refraction, float _roughness) :
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
    CudaMesh(CudaTriangle* _triangles): CudaRTObject(MESH), triangles(_triangles) {}

//    void createOctoTree(Bounds bounds) {
//        float midHorizontal = (bounds.left + bounds.right)/2;
//        float midVertical = (bounds.top + bounds.bottom)/2;
//        float midFrontBack = (bounds.top + bounds.bottom)/2;
//        BVHNode* root = new BVHNode();
//    }
};

class CudaSphere: public CudaRTObject {
public:
    float3 position;
    float radius;
    inline CudaSphere(float3 _position, float _radius, CudaMaterial* _material): CudaRTObject(SPHERE, _material),
                                                                                 position(_position), radius(_radius) {}
};

class CudaScene {
public:
    CudaRTObject** objects;
    int numObjects;
    CudaScene(CudaRTObject** _objects , int _numObjects): objects(_objects), numObjects(_numObjects) {}
};

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------Helper--Functions-------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line);

//template <class T>
//T* cudaWrite(T* data, int len);
//template <class T>
//T* cudaRead(T* data, int len);

float3 vec3ToFloat3(glm::vec3 vec);

Bounds* getNewBounds(std::vector<CudaTriangle*>* triangles);
bool isFloat3InBounds(float3 point, Bounds* bounds);
bool isTriangleInBounds(CudaTriangle* triangle, Bounds* bounds);
BVHBinaryNode* createTreeHelper(std::vector<CudaTriangle*>* localTriangles, BVHBinaryNode* node);

CudaMaterial* materialToCudaMaterial(Material* material);
CudaRTObject* rtObjectToCudaRTObject(RTObject* object);
CudaScene* sceneToCudaScene(Scene* scene);
void cleanCudaScene(CudaScene* scene);
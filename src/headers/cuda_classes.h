#pragma once
#include <glm/glm.hpp>
#include <vector>
#include "bvh_classes.h"
#include "ShaderConst.h"

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------CUDA--OBJECTS-----------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

float3 vec3ToFloat3(glm::vec3 vec);

//----------------------------------------------------------------------------------------------------------------------

class CudaMaterial {
public:
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 reflective;
    float3 transmissive;
    float refraction, roughness, shininess;

    CudaMaterial(float3 _ambient, float3 _diffuse);
    CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float _shininess);
    CudaMaterial(float3 _ambient, float3 _diffuse, float3 _specular, float3 _reflective,
                 float3 _transmissive, float _refraction, float _roughness, float _shininess);
};

//----------------------------------------------------------------------------------------------------------------------

class CudaRTObject {
public:
    int type;
    CudaMaterial* material;
    explicit CudaRTObject(int _type);
    CudaRTObject(int _type, CudaMaterial* _material);
};

//----------------------------------------------------------------------------------------------------------------------

class CudaTriangle {
public:
    float3 a, b, c;
    int index;
    CudaTriangle(float3 _a, float3 _b, float3 _c);
    CudaTriangle(float3 _a, float3 _b, float3 _c, int _index);
};

//----------------------------------------------------------------------------------------------------------------------

class CudaMesh: public CudaRTObject {
public:
    std::vector<CudaTriangle>* hostTriangles;
    CudaTriangle* triangles;
    int numTriangles, maxBVHDepth;
    BVHBinaryNode* bvhRoot;

    CudaMesh();
    explicit CudaMesh(CudaTriangle* _triangles);
    void addTriangle(CudaTriangle _object);
    void finalize();
    BVHBinaryNode* createMeshTree(std::vector<CudaTriangle>* triangles, std::vector<int>* indices, BVHBinaryNode* node);
    static CudaMesh* newHostMesh();
};

//----------------------------------------------------------------------------------------------------------------------

class CudaSphere: public CudaRTObject {
public:
    float3 position;
    float radius;

    CudaSphere(float3 _position, float _radius, CudaMaterial *_material);
};
//----------------------------------------------------------------------------------------------------------------------

class CudaLight {
public:
    int type;
    float3 color;

    CudaLight(int _type);
    CudaLight(int _type, float3 _color);
};

//----------------------------------------------------------------------------------------------------------------------

class CudaSkyboxLight: public CudaLight {
public:
    int sphereIndex; // Index of the sphere used for ray-hit detection

    CudaSkyboxLight();
    CudaSkyboxLight(int _sphereIndex);
};

//----------------------------------------------------------------------------------------------------------------------

class CudaPointLight: public CudaLight {
public:
    float3 position;
    CudaPointLight(float3 _position);
    CudaPointLight(float3 _position, float3 _color);
};

//----------------------------------------------------------------------------------------------------------------------


class CudaMeshLight: public CudaLight {
public:
    CudaMeshLight();
};

//----------------------------------------------------------------------------------------------------------------------
class CudaScene {
public:
    std::vector<CudaRTObject*>* hostObjects;
    std::vector<CudaLight*>* hostLights;
    CudaRTObject** objects;
    CudaLight** lights;
    int numObjects, numLights;

    CudaScene();
    CudaScene(CudaRTObject** _objects , int _numObjects);
    CudaScene(CudaRTObject** _objects , int _numObjects, CudaLight** _lights , int _numLights);
    void addObject(CudaRTObject* _object);
    void addLight(CudaLight* _light);
    static CudaScene* newHostScene();
};

//----------------------------------------------------------------------------------------------------------------------

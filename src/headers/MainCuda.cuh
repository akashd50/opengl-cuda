#pragma once
#include "Texture.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>
#include <vector_types.h>
#include <vector>

const int SPHERE = 1;
const int MESH = 2;
const int PLANE = 3;

#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

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
    inline RTObject( int _type) : type (_type) {}

    inline void setMaterial(Material* _material) {
        RTObject::material = _material;
    }

    inline Material* getMaterial() {
        return material;
    }

    inline int getType() {
        return type;
    }

    inline ~RTObject() {
        delete material;
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
    CudaMaterial material;
    inline CudaRTObject(int _type) : type(_type) {}
    inline CudaRTObject(int _type, CudaMaterial _material) : type(_type), material(_material) {}
};

class CudaSphere: public CudaRTObject {
public:
    float3 position;
    float radius;
    inline CudaSphere(float3 _position, float _radius, CudaMaterial _material) : CudaRTObject(SPHERE, _material),
    position(_position), radius(_radius) {}
};

class CudaScene {
public:
    CudaRTObject** objects;
    int numObjects;
    CudaScene(CudaRTObject** _objects , int _numObjects): objects(_objects), numObjects(_numObjects) {}
};

float3 vec3ToFloat3(glm::vec3 vec);
CudaMaterial materialToCudaMaterial(Material* material);
CudaRTObject* rtObjectToCudaRTObject(RTObject* object);
CudaScene* sceneToCudaScene(Scene* scene);

class MainCuda {
public:
    static void renderRayTracedScene(Texture* texture, Scene* scene);
    static void doCalculation();
};
#include "headers/cuda_classes.h"

//CudaMaterial* CudaMaterialBuilder::getMaterial() {
//    return new CudaMaterial(ambient, diffuse, specular, reflective,
//                            transmissive, refraction, roughness, shininess);
//}

float3 vec3ToFloat3(glm::vec3 vec) {
    return make_float3(vec.x, vec.y, vec.z);
}
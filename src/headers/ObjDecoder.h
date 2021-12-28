#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "headers/cuda_classes.h"

struct config {
    int v1, v2, v3;
    int n1, n2, n3;
    int t1, t2, t3;
};

struct RawData {
    std::vector<glm::vec3>* vertices;
    std::vector<glm::vec2>* uvs;
    std::vector<config>* faceConfiguration;
};

class ObjDecoder {
public:
    static glm::mat4 createTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    static CudaMesh* createMesh(const std::string& file, glm::mat4 transformationMatrix);
    static RawData readFile(const std::string& filename);
};

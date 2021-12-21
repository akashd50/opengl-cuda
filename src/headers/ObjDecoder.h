#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "headers/CudaUtils.cuh"

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
    static Mesh* createMesh(const std::string& file);
    static RawData readFile(const std::string& filename);
};

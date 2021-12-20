#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;

class ObjDecoder {
    struct config {
        int v1, v2, v3;
        int n1, n2, n3;
        int t1, t2, t3;
    };

private:
    std::vector<glm::vec3>* vertices;
    std::vector<glm::vec3>* normals;
    std::vector<glm::vec2>* uvs;
    std::vector<config>* faceConfiguration;

public:
    ObjDecoder(string file);
    void readFile(string filename);
};
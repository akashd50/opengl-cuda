#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "headers/ObjDecoder.h"
#include "headers/Utils.h"

using namespace std;

ObjDecoder::ObjDecoder(string file) {
    readFile(file);
    float* localVerts = new float[faceConfiguration->size() * 3 * 3];
    int vertsIndex = 0;
    for (int i = 0; i < faceConfiguration->size(); i++) {
        config curr = faceConfiguration->at(i);

        localVerts[vertsIndex++] = (float)vertices->at(curr.v1).x;
        localVerts[vertsIndex++] = (float)vertices->at(curr.v1).y;
        localVerts[vertsIndex++] = (float)vertices->at(curr.v1).z;

        localVerts[vertsIndex++] = (float)vertices->at(curr.v2).x;
        localVerts[vertsIndex++] = (float)vertices->at(curr.v2).y;
        localVerts[vertsIndex++] = (float)vertices->at(curr.v2).z;

        localVerts[vertsIndex++] = (float)vertices->at(curr.v3).x;
        localVerts[vertsIndex++] = (float)vertices->at(curr.v3).y;
        localVerts[vertsIndex++] = (float)vertices->at(curr.v3).z;
    }

    cout << "End of Loading" << "\n";

    delete[] localVerts;

    delete vertices;
    delete normals;
    delete uvs;
}

void ObjDecoder::readFile(string filename) {
    vertices = new vector<glm::vec3>;
    normals = new vector<glm::vec3>;
    uvs = new vector<glm::vec2>;
    faceConfiguration = new vector<config>;

    ifstream dataFile;
    dataFile.open(filename);

    string line;
    if (dataFile.is_open()) {
        cout << "File Opened! Reading now..." << "\n";

        while (getline(dataFile, line)) {
            vector<string>* lineTokens = Utils::tokenize(line, " ");

            if (lineTokens->at(0) == "v") {
                glm::vec3 vert = glm::vec3(stof(lineTokens->at(1)), stof(lineTokens->at(2)), stof(lineTokens->at(3)));
                vertices->push_back(vert);
            }
            else if (lineTokens->at(0) == "vt") {
                glm::vec2 uv = glm::vec2(stof(lineTokens->at(1)), stof(lineTokens->at(2)));
                uvs->push_back(uv);
            }
            else if (lineTokens->at(0) == "f") {
                vector<string>* faceV1Tokens = Utils::tokenize(lineTokens->at(1), "/");
                vector<string>* faceV2Tokens = Utils::tokenize(lineTokens->at(2), "/");
                vector<string>* faceV3Tokens = Utils::tokenize(lineTokens->at(3), "/");

                config c;
                c.v1 = stoi(faceV1Tokens->at(0)) - 1;
                c.v2 = stoi(faceV2Tokens->at(0)) - 1;
                c.v3 = stoi(faceV3Tokens->at(0)) - 1;

                c.t1 = stoi(faceV1Tokens->at(1)) - 1;
                c.t2 = stoi(faceV2Tokens->at(1)) - 1;
                c.t3 = stoi(faceV3Tokens->at(1)) - 1;

                c.n1 = stoi(faceV1Tokens->at(2)) - 1;
                c.n2 = stoi(faceV2Tokens->at(2)) - 1;
                c.n3 = stoi(faceV3Tokens->at(2)) - 1;

                faceConfiguration->push_back(c);
            }
            else if (lineTokens->at(0) == "vn") {
                glm::vec3 normal = glm::vec3(stof(lineTokens->at(1)), stof(lineTokens->at(2)), stof(lineTokens->at(3)));
                normals->push_back(normal);
            }

            delete lineTokens;
        }
        dataFile.close();
    }
    else {
        cout << "Error encountered during opening file: " + filename << "\n";
    }
}
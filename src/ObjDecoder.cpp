#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "headers/ObjDecoder.h"
#include "headers/Utils.h"

std::vector<Triangle*>* ObjDecoder::getTriangles(const std::string& file) {
    RawData rawData = readFile(file);
    auto triangles = new std::vector<Triangle*>;
    for (int i = 0; i < rawData.faceConfiguration->size(); i++) {
        config curr = rawData.faceConfiguration->at(i);
        glm::vec3 v1 = rawData.vertices->at(curr.v1);
        glm::vec3 v2 = rawData.vertices->at(curr.v2);
        glm::vec3 v3 = rawData.vertices->at(curr.v3);
        triangles->push_back(new Triangle(v1, v2, v3));
    }

    std::cout << "End of Loading" << "\n";

    delete rawData.vertices;
    delete rawData.uvs;
    delete rawData.faceConfiguration;
    return triangles;
}

RawData ObjDecoder::readFile(const std::string& filename) {
    RawData rawData;
    rawData.vertices = new std::vector<glm::vec3>;
    rawData.uvs = new std::vector<glm::vec2>;
    rawData.faceConfiguration = new std::vector<config>;

    std::ifstream dataFile;
    dataFile.open(filename);

    std::string line;
    if (dataFile.is_open()) {
        std::cout << "File Opened! Reading now..." << "\n";

        while (getline(dataFile, line)) {
            std::vector<std::string>* lineTokens = Utils::tokenize(line, " ");

            if (lineTokens->at(0) == "v") {
                glm::vec3 vert = glm::vec3(stof(lineTokens->at(1)), stof(lineTokens->at(2)), stof(lineTokens->at(3)));
                rawData.vertices->push_back(vert);
            }
            else if (lineTokens->at(0) == "vt") {
                glm::vec2 uv = glm::vec2(stof(lineTokens->at(1)), stof(lineTokens->at(2)));
                rawData.uvs->push_back(uv);
            }
            else if (lineTokens->at(0) == "f") {
                std::vector<std::string>* faceV1Tokens = Utils::tokenize(lineTokens->at(1), "/");
                std::vector<std::string>* faceV2Tokens = Utils::tokenize(lineTokens->at(2), "/");
                std::vector<std::string>* faceV3Tokens = Utils::tokenize(lineTokens->at(3), "/");

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

                rawData.faceConfiguration->push_back(c);
            }
            delete lineTokens;
        }
        dataFile.close();
    }
    else {
        std::cout << "Error encountered during opening file: " + filename << "\n";
    }
    return rawData;
}
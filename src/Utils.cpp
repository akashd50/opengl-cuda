#include "headers/Utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "headers/stb_image.h"

std::string Utils::readTextFile(const std::string& filename) {
    std::ifstream dataFile;
    dataFile.open(filename);

    std::string shaderCode;
    std::string line;
    if (dataFile.is_open()) {
        while (getline(dataFile, line)) {
            shaderCode += line + "\n";
        }
        dataFile.close();
    }
    else {
        std::cout << "Error when opening file: " + filename << "\n";
    }
    return shaderCode;
}

ImageData* Utils::readImageFile(std::string filename) {
    auto toReturn = new ImageData();
    stbi_set_flip_vertically_on_load(true);
    uint8_t* image_data = stbi_load(filename.c_str(),
        &toReturn->width, &toReturn->height, &toReturn->numChannels, 3);
    const char* failReason = stbi_failure_reason();
    toReturn->pixelData = image_data;
    return toReturn;
}

void Utils::free(ImageData* imageData) {
    stbi_image_free(imageData->pixelData);
}

std::vector<std::string>* Utils::tokenize(std::string line, std::string del) {
    char* context = nullptr;
    auto tokens = new std::vector<std::string>;
    char* cStyleLine = (char*)line.c_str();
    char* token = strtok_s(cStyleLine, del.c_str(), &context);
    while (token != nullptr) {
        tokens->push_back(token);
        token = strtok_s(nullptr, del.c_str(), &context);
    }
    return tokens;
}
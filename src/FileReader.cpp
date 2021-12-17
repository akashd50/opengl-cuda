#include "headers/FileReader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "headers/stb_image.h"

std::string FileReader::readTextFile(std::string filename) {
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

ImageData* FileReader::readImageFile(std::string filename) {
    ImageData* toReturn = new ImageData();
    stbi_set_flip_vertically_on_load(true);
    uint8_t* image_data = stbi_load(filename.c_str(),
        &toReturn->width, &toReturn->height, &toReturn->numChannels, 3);
    const char* failReason = stbi_failure_reason();
    toReturn->pixelData = image_data;
    return toReturn;
}

void FileReader::free(ImageData* imagedata) {
    stbi_image_free(imagedata->pixelData);
}
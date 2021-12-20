#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <string>

struct ImageData {
	unsigned char* pixelData;
	int width, height, numChannels;
};

//----------------------------------------------------------------------------------------------------------------------
class Utils {
public:
    static std::string readTextFile(const std::string& filename);
    static ImageData* readImageFile(std::string filename);
    static void free(ImageData* imageData);
    static std::vector<std::string>* tokenize(std::string line, std::string del);
};

//----------------------------------------------------------------------------------------------------------------------


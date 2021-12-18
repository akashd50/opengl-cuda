#pragma once
#include <iostream>
#include <string>
#include <fstream>

struct ImageData {
	unsigned char* pixelData;
	int width, height, numChannels;
};

class FileReader {
public:
    static std::string readTextFile(const std::string& filename);
	static ImageData* readImageFile(std::string filename);
	static void free(ImageData* imagedata);
};

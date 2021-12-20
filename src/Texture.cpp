#include "headers/Texture.h"
#include "headers/Utils.h"

GLuint getColorFormat(int numChannels) {
	switch (numChannels)
	{
		case 3: return GL_RGB;
		case 4: return GL_RGBA;
		default: return GL_RGB;
	}
}

Texture::Texture(int width, int height, GLuint format) {
	this->width = width;
	this->height = height;

	//glCreateTextures(GL_TEXTURE_2D, 1, &texture_id);
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, format, this->width, this->height, 0, format, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::Texture(std::string filename) {
	ImageData* data = Utils::readImageFile(filename);
	this->width = data->width;
	this->height = data->height;
	this->numChannels = data->numChannels;

	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	const unsigned char* imaged = data->pixelData;
	
	if (data->pixelData) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, getColorFormat(numChannels),
			GL_UNSIGNED_BYTE, data->pixelData);

		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else {
		std::cout << "ERROR: Unable to load texture" << std::endl;
	}

	glBindTexture(GL_TEXTURE_2D, 0);

    Utils::free(data);
}

Texture::~Texture() {
	glDeleteTextures(1, &texture_id);
}

GLuint Texture::getTextureId() {
	return texture_id;
}

void Texture::setTextureId(GLuint t_id) {
	this->texture_id = t_id;
}

float Texture::getRatio() {
	return this->width / this->height;
}

int Texture::getWidth() {
	return this->width;
}

int Texture::getHeight() {
	return this->height;
}

void Texture::setWidth(int w) {
	this->width = w;
}

void Texture::setHeight(int h) {
	this->height = h;
}

void Texture::setNumChannels(int c) {
	this->numChannels = c;
}

int Texture::getNumChannels() {
	return this->numChannels;
}
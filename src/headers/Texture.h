#pragma once
#include <iostream>
#include <string>
#include <GL/glew.h>

class Texture {
private:
	GLuint texture_id;
	int width, height, numChannels;
public:
	Texture(int width, int height, GLuint format);
	Texture(std::string filename);
	~Texture();

	GLuint getTextureId();

	void setTextureId(GLuint t_id);

	float getRatio();

	int getWidth();

	int getHeight();

	void setWidth(int w);

	void setHeight(int h);

	void setNumChannels(int c);

	int getNumChannels();
};
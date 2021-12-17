#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

class Framebuffer {
private:
	int width, height;
	GLuint framebufferObject, textureAttachment, textureAttachment2, renderBufferAttachment;
public:
	Framebuffer(int w, int h);

	void addTextureBuffer();
	void addSecondTextureBuffer();
	void addRenderBuffer();

	void bind(bool isFirst);
	void bindRead();
	void unbind();

	GLuint getTextureAttachment();

	GLuint getFrameBuffer();

	GLuint getTextureAttachment2();
	
	void reshape(int w, int h);
};


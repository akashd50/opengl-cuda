#include "headers/Framebuffer.h"
#include <iostream>

Framebuffer::Framebuffer(int w, int h) {
	width = w; height = h;

	// generate and bind a new framebuffer
	glGenFramebuffers(1, &framebufferObject);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);

	// add texture and depth attachments
	addTextureBuffer();
	addRenderBuffer();

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "FRAMEBUFFER INITIALIZED" << std::endl;
	}
	else {
		std::cout << "FRAMEBUFFER FAILED" << std::endl;
	}

	unbind();
}

void Framebuffer::addTextureBuffer() {
	// create and bind new texture
	glGenTextures(1, &textureAttachment);
	glBindTexture(GL_TEXTURE_2D, textureAttachment);

	// set texture formating and parameters
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureAttachment, 0);
	//glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, textureAttachment, 0);
}

void Framebuffer::addSecondTextureBuffer() {
	// create and bind new texture
	glGenTextures(1, &textureAttachment2);
	glBindTexture(GL_TEXTURE_2D, textureAttachment2);

	// set texture formating and parameters
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureAttachment2, 0);
}

void Framebuffer::addRenderBuffer() {
	glGenRenderbuffers(1, &renderBufferAttachment);
	glBindRenderbuffer(GL_RENDERBUFFER, renderBufferAttachment);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferAttachment);
	//glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, renderBufferAttachment, 0);
}

void Framebuffer::bind(bool isFirst) {
	// bind framebuffer adn set the viewport width and height 
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferObject);
	glViewport(0, 0, width, height);
	/*if (isFirst) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.1f, 0.2f, 0.1f, 1.0f);
	}*/
}

void Framebuffer::bindRead() {
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferObject);
}

void Framebuffer::unbind() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint Framebuffer::getTextureAttachment() {
	return this->textureAttachment;
}

GLuint Framebuffer::getFrameBuffer() {
	return this->framebufferObject;
}

GLuint Framebuffer::getTextureAttachment2() {
	return this->textureAttachment2;
}

void Framebuffer::reshape(int w, int h) {
	width = w;
	height = h;

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebufferObject);

	// unbind framebuffer and renderbuffer attachments
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);

	// delete old texture and render buffer attachments
	glDeleteTextures(1, &textureAttachment);
	glDeleteRenderbuffers(1, &renderBufferAttachment);

	// create new attachments based on new width and height
	addTextureBuffer();
	addRenderBuffer();

	//unbind framebuffer
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}
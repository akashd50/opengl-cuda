#pragma once
#include "MainOpenGL.h"
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "MainCuda.h"
#define M_PI 3.14159265358979323846264338327950288

//----------------------------------------------------------------------------
//Quad* quad;
//Shader* single_color_shader, *texture_shader;
//Framebuffer* default_framebuffer;
//Texture* test_texture;

glm::vec3 s(int x, int y) {
    float d = 1.0;
    float fov = 60.0;
    float aspect_ratio = ((float)MainOpenGL::WIDTH) / ((float)MainOpenGL::HEIGHT);
    float h = d * (float)tan((M_PI * fov) / 180.0 / 2.0);
    float w = h * aspect_ratio;

    float top = h;
    float bottom = -h;
    float left = -w;
    float right = w;

    float u = left + (right - left) * (x) / ((float)MainOpenGL::WIDTH);
    float v = bottom + (top - bottom) * (((float)MainOpenGL::HEIGHT) - y) / ((float)MainOpenGL::HEIGHT);

    return glm::vec3(u, v, -d);
}

// -------------------------------------------------------------------------------------------------------
const char* MainOpenGL::WINDOW_TITLE = "Raytracing with Cuda";
const double MainOpenGL::FRAME_RATE_MS = 1000.0 / 60.0;
int MainOpenGL::WIDTH = 512;
int MainOpenGL::HEIGHT = 512;
// -------------------------------------------------------------------------------------------------------
void MainOpenGL::init()
{
    //default_framebuffer = new Framebuffer(600, 600);
//    single_color_shader = new Shader("shaders/single_color_vs.glsl", "shaders/single_color_fs.glsl");
//    texture_shader = new Shader("shaders/texture_vs.glsl", "shaders/texture_fs.glsl");
//    // test_texture = new Texture("res/scr1.bmp");
//    test_texture = new Texture(800, 800, GL_RGBA);
//
//    quad = new Quad();
//    //quad->build(single_color_shader);
//    quad->setTexture(test_texture);
//    quad->build(texture_shader);
//
//    MainOpenCL::setup(quad);
//    glFinish();
//    MainOpenCL::onDrawFrame();
    MainCuda::doCalculation();

    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0, 1.0, 1.0, 1.0);
}

//----------------------------------------------------------------------------

void MainOpenGL::display(void)
{
    //default_framebuffer->bind(true);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0, 0.5, 0.5, 1.0);

//    quad->getShader()->useProgram();
//    quad->applyTransformations();
//    quad->onDrawFrame();
//
//    glFinish();
    // MainOpenCL::onDrawFrame();
    //default_framebuffer->unbind();

    glutSwapBuffers();
    //glutPostRedisplay();
}

//----------------------------------------------------------------------------

void MainOpenGL::keyboard(unsigned char key, int x, int y)
{
    switch (key) {
    case 033: // Escape Key
    case 'q': case 'Q':
        exit(EXIT_SUCCESS);
        break;
    }
}

//----------------------------------------------------------------------------

void MainOpenGL::mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
    }
}

//----------------------------------------------------------------------------

void MainOpenGL::update(void)
{

}

//----------------------------------------------------------------------------

void MainOpenGL::reshape(int width, int height)
{
    WIDTH = width;
    HEIGHT = height;
    glViewport(0, 0, width, height);
    //default_framebuffer->reshape(width, height);
    /*GLfloat aspect = GLfloat(width) / height;
    glm::mat4  projection = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 3.0f);*/
}

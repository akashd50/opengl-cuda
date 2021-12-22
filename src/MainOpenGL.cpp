#pragma once
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "headers/CudaUtils.cuh"
#include "headers/MainOpenGL.h"
#include "headers/Quad.h"
#include "headers/ObjDecoder.h"

#define M_PI 3.14159265358979323846264338327950288

//----------------------------------------------------------------------------
CudaUtils* cudaUtils;
CudaScene* cudaScene;
Quad* quad;
Shader* single_color_shader, *texture_shader;
//Framebuffer* default_framebuffer;
Texture* test_texture;
Scene* scene;

// -------------------------------------------------------------------------------------------------------
const char* MainOpenGL::WINDOW_TITLE = "Raytracing with Cuda";
const double MainOpenGL::FRAME_RATE_MS = 1000.0 / 60.0;
int MainOpenGL::WIDTH = 512;
int MainOpenGL::HEIGHT = 512;
// -------------------------------------------------------------------------------------------------------
void MainOpenGL::init()
{
    //default_framebuffer = new Framebuffer(600, 600);
    //single_color_shader = new Shader("shaders/single_color_vs.glsl", "shaders/single_color_fs.glsl");
    texture_shader = new Shader("../shaders/texture_vs.glsl", "../shaders/texture_fs.glsl");
    //test_texture = new Texture("../resources/scr1.bmp");
    test_texture = new Texture(512, 512, GL_RGBA);

    quad = new Quad();
    //quad->build(single_color_shader);
    quad->setTexture(test_texture);
    quad->build(texture_shader);

    scene = new Scene();
    auto mat1 = new Material(glm::vec3(0.1), glm::vec3(0.1, 0.5, 0.4),
                                  glm::vec3(1.0), 1.0, glm::vec3(0.4),
                                  glm::vec3(0.2), 1.0);
    auto mat2 = new Material(glm::vec3(0.1), glm::vec3(0.6, 0.2, 0.1),
                             glm::vec3(1.0), 1.0, glm::vec3(0.3),
                             glm::vec3(0.2), 1.0);

    scene->addObject(new Sphere(mat1, 2.0, glm::vec3(2.0, 0.0, -7.0)));
    scene->addObject(new Sphere(mat2,0.5, glm::vec3(-1.0, 0.0, -4.0)));

    Mesh* mesh = ObjDecoder::createMesh("../resources/cube.obj");
    mesh->setMaterial(mat2);
    scene->addObject(mesh);

    cudaScene = allocateCudaScene(scene);

    cudaUtils = new CudaUtils();
    cudaUtils->deviceInformation();
    cudaUtils->initializeRenderSurface(test_texture);
    cudaUtils->renderScene(cudaScene);

    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0, 1.0, 1.0, 1.0);
}

//----------------------------------------------------------------------------

void MainOpenGL::display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0, 0.5, 0.5, 1.0);

    quad->getShader()->useProgram();
    quad->applyTransformations();
    quad->onDrawFrame();

    glutSwapBuffers();
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
        cudaUtils->onClick(x, y, cudaScene);
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

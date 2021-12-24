#pragma once
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "headers/CudaKernelUtils.cuh"
#include "headers/MainOpenGL.h"
#include "headers/Quad.h"
#include "headers/ObjDecoder.h"

#define M_PI 3.14159265358979323846264338327950288

//----------------------------------------------------------------------------
CudaKernelUtils* cudaUtils;
CudaScene* cudaScene, *allocatedScene;
Quad* quad;
Shader* single_color_shader, *texture_shader;
//Framebuffer* default_framebuffer;
Texture* test_texture;
// ---------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------------
const char* MainOpenGL::WINDOW_TITLE = "Raytracing with Cuda";
const double MainOpenGL::FRAME_RATE_MS = 1000.0 / 60.0;
int MainOpenGL::WIDTH = 512;
int MainOpenGL::HEIGHT = 512;
// ---------------------------------------------------------------------------------------------------------------------
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

    cudaScene = CudaScene::newHostScene();

    auto mat1 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.1, 0.5, 0.4));
    mat1->reflective = make_float3(0.4, 0.4, 0.4);

    auto mat3 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.7, 0.3, 0.2));
    mat3->reflective = make_float3(0.4, 0.4, 0.4);

    cudaScene->addObject(new CudaSphere(make_float3(3.0, 0.0, -7.0), 2.0, mat1));
    //scene->addObject(new Sphere(mat2,0.5, glm::vec3(-1.0, 0.0, -4.0)));

    CudaMesh* mesh = ObjDecoder::createMesh("../resources/cylinder.obj");
    mesh->material = mat3;
    cudaScene->addObject(mesh);

    allocatedScene = allocateCudaScene(cudaScene);

    cudaUtils = new CudaKernelUtils();
    cudaUtils->deviceInformation();
    cudaUtils->initializeRenderSurface(test_texture);
    cudaUtils->renderScene(allocatedScene);

    std::cout << "Size of int: " << sizeof(int) << std::endl;
    std::cout << "Size of pointer: " << sizeof(CudaMesh*) << std::endl;

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
        cudaUtils->onClick(x, y, allocatedScene);
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

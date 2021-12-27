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
int currRowIndex, currColumnIndex;

// ---------------------------------------------------------------------------------------------------------------------
const char* MainOpenGL::WINDOW_TITLE = "Raytracing with Cuda";
const double MainOpenGL::FRAME_RATE_MS = 1000.0 / 60.0;
const int RENDER_BLOCK_SIZE = 32;
const int MAX_RENDER_THREADS_SIZE = 256;

int MainOpenGL::WIDTH = 768;
int MainOpenGL::HEIGHT = 768;
// ---------------------------------------------------------------------------------------------------------------------
void MainOpenGL::init()
{
    //default_framebuffer = new Framebuffer(600, 600);
    //single_color_shader = new Shader("shaders/single_color_vs.glsl", "shaders/single_color_fs.glsl");
    texture_shader = new Shader("../shaders/texture_vs.glsl", "../shaders/texture_fs.glsl");
    //test_texture = new Texture("../resources/scr1.bmp");
    test_texture = new Texture(MainOpenGL::WIDTH, MainOpenGL::HEIGHT, GL_RGBA);

    quad = new Quad();
    //quad->build(single_color_shader);
    quad->setTexture(test_texture);
    quad->build(texture_shader);

    cudaScene = CudaScene::newHostScene();
    cudaScene->width = MainOpenGL::WIDTH;
    cudaScene->height = MainOpenGL::HEIGHT;

    cudaScene->addLight(new CudaSkyboxLight(new CudaSphere(make_float3(0.0, 0.0, 0.0), 50.0)));

    auto mat1 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.1, 0.6, 0.1));
    mat1->reflective = make_float3(0.4, 0.4, 0.4);
    mat1->roughness = 1.0f;

    auto mat2 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.6, 0.6, 0.6));
    mat2->reflective = make_float3(0.4, 0.4, 0.4);
    mat2->roughness = 1.0f;

    auto mat3 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.3, 0.2, 0.6));
    mat3->reflective = make_float3(0.4, 0.4, 0.4);
    mat3->roughness = 1.0f;

    cudaScene->addObject(new CudaSphere(make_float3(3.0, 0.0, -7.0), 2.0, mat1));

    CudaMesh* mesh = ObjDecoder::createMesh("../resources/monkey_mid.obj");
    mesh->material = mat3;
    cudaScene->addObject(mesh);

    CudaMesh* floor = ObjDecoder::createMesh("../resources/floor.obj");
    floor->material = mat2;
    cudaScene->addObject(floor);

    allocatedScene = allocateCudaScene(cudaScene);

    cudaUtils = new CudaKernelUtils();
    cudaUtils->deviceInformation();
    cudaUtils->initializeRenderSurface(test_texture);
    //cudaUtils->renderScene(allocatedScene, 512, cudaScene->width, 0);

    currRowIndex = 0;
    currColumnIndex = 0;

    std::cout << "Size of int: " << sizeof(int) << std::endl;
    std::cout << "Size of pointer: " << sizeof(CudaMaterial*) << std::endl;
    std::cout << "Size of Sphere: " << sizeof(CudaSphere) << std::endl;
    std::cout << "Size of float3: " << sizeof(float3) << std::endl;
    std::cout << "Size of float: " << sizeof(float) << std::endl;
    std::cout << "Size of CudaRTObject: " << sizeof(CudaRTObject) << std::endl;

    std::cout << "Size of Sphere + RTObject: " << sizeof(CudaSphere) + sizeof(CudaRTObject) << std::endl;

    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0, 1.0, 1.0, 1.0);
}

//----------------------------------------------------------------------------

void MainOpenGL::display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0, 0.5, 0.5, 1.0);

    if (currRowIndex < cudaScene->height) {
        if (currColumnIndex < cudaScene->width) {
            int numThreadsToRun = std::min(cudaScene->width - currColumnIndex, MAX_RENDER_THREADS_SIZE);
            cudaUtils->renderScene(allocatedScene, RENDER_BLOCK_SIZE, numThreadsToRun, currRowIndex, currColumnIndex);
            currColumnIndex += MAX_RENDER_THREADS_SIZE;
        } else {
            currColumnIndex = 0;
            currRowIndex += RENDER_BLOCK_SIZE;
        }
    }

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
        cleanCudaScene(allocatedScene);
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

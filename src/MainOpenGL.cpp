#pragma once
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <chrono>
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

const char* MainOpenGL::WINDOW_TITLE = "Raytracing with Cuda";
const double MainOpenGL::FRAME_RATE_MS = 1000.0 / 60.0;
const int RENDER_BLOCK_SIZE = 32;
const int MAX_RENDER_THREADS_SIZE = 256;
const int MAX_SAMPLES_PER_PIXEL = 64;

int MainOpenGL::WIDTH = 768;
int MainOpenGL::HEIGHT = 768;

// ---------------------------------------------------------------------------------------------------------------------
int currRowIndex, currColumnIndex, sampleIndex;
bool clockStart, clockEnd, runDenoiseKernel;
std::chrono::time_point<std::chrono::steady_clock> renderTimeStart;
// ---------------------------------------------------------------------------------------------------------------------

void MainOpenGL::init()
{
    texture_shader = new Shader("../shaders/texture_vs.glsl", "../shaders/texture_fs.glsl");
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
    mat1->reflective = make_float3(0.2, 0.2, 0.2);
    mat1->albedo = 0.5;
    mat1->roughness = 0.8f;

    auto mat2 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.6, 0.6, 0.6));
    mat2->reflective = make_float3(0.3, 0.3, 0.3);
    mat2->albedo = 0.5;
    mat2->roughness = 0.5f;

    auto mat3 = new CudaMaterial(make_float3(0.1, 0.1, 0.1), make_float3(0.3, 0.2, 0.6));
    mat3->reflective = make_float3(0.2, 0.2, 0.2);
    mat3->albedo = 0.5;
    mat3->roughness = 1.0f;

    cudaScene->addObject(new CudaSphere(make_float3(2.0, -1.0, -8.0), 2.0, mat1));
    glm::mat4 meshLightT = ObjDecoder::createTransformationMatrix(glm::vec3(0.0, 3.0f, -5.0f),
                                                             glm::vec3( 0.0f, 0.0f, 0.0f),
                                                             glm::vec3(1));
    auto meshLightObj = ObjDecoder::createMesh("../resources/cube.obj", meshLightT);
    auto meshLight = new CudaMeshLight(meshLightObj, make_float3(0.0, 1.0, 1.0));
    meshLight->intensity = 5.0f;
    cudaScene->addLight(meshLight);

    glm::mat4 meshT = ObjDecoder::createTransformationMatrix(glm::vec3(-1.0, 0.0f, -7.0f),
                                                             glm::vec3( 0.0f, 180.0f, 0.0f),
                                                             glm::vec3(1));
    CudaMesh* mesh = ObjDecoder::createMesh("../resources/monkey_mid.obj", meshT);
    mesh->material = mat3;
    cudaScene->addObject(mesh);

//    glm::mat4 floorT = ObjDecoder::createTransformationMatrix(glm::vec3(0, 0.0f, -9.0f),
//                                                              glm::vec3( 0.0f, 0.0f, 0.0f),
//                                                              glm::vec3(5.0f, 4.0f, 10.0f));
//
    glm::mat4 floorT = ObjDecoder::createTransformationMatrix(glm::vec3(0, -3.2f, 0.0f),
                                                                glm::vec3( 0.0f, 0.0f, 0.0f),
                                                                glm::vec3(10.0f, 0.2f, 10.0f));
    CudaMesh* floor = ObjDecoder::createMesh("../resources/room_closed.obj", floorT);
    floor->material = mat2;
    cudaScene->addObject(floor);

    allocatedScene = allocateCudaScene(cudaScene);

    cudaUtils = new CudaKernelUtils();
    cudaUtils->deviceInformation();
    cudaUtils->initializeRenderSurface(test_texture);
    //cudaUtils->renderScene(allocatedScene, 512, cudaScene->width, 0);

    currRowIndex = 0;
    currColumnIndex = 0;
    sampleIndex = 0;
    clockStart = false;
    clockEnd = false;

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
        if (!clockStart) {
            renderTimeStart = std::chrono::steady_clock::now();
            clockStart = true;
        }

        if (currColumnIndex < cudaScene->width) {
            int numThreadsToRun = std::min(cudaScene->width - currColumnIndex, MAX_RENDER_THREADS_SIZE);
            if (runDenoiseKernel) {
                cudaUtils->runDenoiseKernel(cudaScene, RENDER_BLOCK_SIZE, numThreadsToRun, currRowIndex, currColumnIndex, sampleIndex);
            } else {
                cudaUtils->renderScene(allocatedScene, RENDER_BLOCK_SIZE, numThreadsToRun, currRowIndex, currColumnIndex, sampleIndex);
            }
            currColumnIndex += MAX_RENDER_THREADS_SIZE;
        } else {
            currColumnIndex = 0;
            currRowIndex += RENDER_BLOCK_SIZE;
        }
    } else {
        if (sampleIndex < MAX_SAMPLES_PER_PIXEL) {
            sampleIndex++;
            //runDenoiseKernel = !runDenoiseKernel;
            currColumnIndex = 0;
            currRowIndex = 0;
            std::cout << "Next Samples Index.. " << sampleIndex << std::endl;
        }
        if (!clockEnd) {
            auto renderTimeEnd = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = renderTimeEnd - renderTimeStart;
            std::cout << "\nElapsed time: " << elapsed_seconds.count() << "s\n";
            clockEnd = true;
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
        case 'r':
        case 'R':
            std::cout << "Running Another Sample..." << std::endl;
            clockStart = false;
            clockEnd = false;
            sampleIndex++;
            currRowIndex = 0;
            currColumnIndex = 0;
            runDenoiseKernel = false;
            break;
        case 'd':
        case 'D':
            std::cout << "Running DeNoiser..." << std::endl;
            currRowIndex = 0;
            currColumnIndex = 0;
            runDenoiseKernel = true;
            //cudaUtils->runDenoiseKernel(allocatedScene, WIDTH, HEIGHT, 0, 0);
            break;
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

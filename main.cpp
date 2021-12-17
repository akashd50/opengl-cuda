#pragma once
#include <iostream>
#include <math.h>
#include "glm/glm.hpp"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
//__global__
//void add(int n, float *x, float *y)
//{
//    int index = threadIdx.x;
//    int stride = blockDim.x;
//    for (int i = index; i < n; i+=stride)
//        y[i] = x[i] + y[i];
//}
//
//int main(void)
//{
//    glm::vec3 a;
//    a.x = 3;
//    std::cout << "A is: " << a.x << std::endl;
//
//    int N = 1<<20; // 1M elements
//
//    // Allocate Unified Memory -- accessible from CPU or GPU
//    float *x, *y;
//    cudaMallocManaged(&x, N*sizeof(float));
//    cudaMallocManaged(&y, N*sizeof(float));
//
//    // initialize x and y arrays on the host
//    for (int i = 0; i < N; i++) {
//        x[i] = 1.0f;
//        y[i] = 2.0f;
//    }
//
//    // Run kernel on 1M elements on the GPU
//    add<<<1, 256>>>(N, x, y);
//    cudaDeviceSynchronize();
//
//    // Check for errors (all values should be 3.0f)
//    float maxError = 0.0f;
//    for (int i = 0; i < N; i++)
//        maxError = fmax(maxError, fabs(y[i]-3.0f));
//    std::cout << "Max error: " << maxError << std::endl;
//
//    // Free memory
//    cudaFree(x);
//    cudaFree(y);
//
//    return 0;
//}

#include "MainOpenGL.h"


#define MAX_SOURCE_SIZE (0x100000)

void timer(int unused)
{
    MainOpenGL::update();
    glutPostRedisplay();
    glutTimerFunc(MainOpenGL::FRAME_RATE_MS, timer, 0);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(MainOpenGL::WIDTH, MainOpenGL::HEIGHT);
    glutInitContextVersion(3, 2);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutCreateWindow(MainOpenGL::WINDOW_TITLE);

    glewInit();

    MainOpenGL::init();

    glutDisplayFunc(MainOpenGL::display);
    glutKeyboardFunc(MainOpenGL::keyboard);
    glutMouseFunc(MainOpenGL::mouse);
    glutReshapeFunc(MainOpenGL::reshape);
    glutTimerFunc(MainOpenGL::FRAME_RATE_MS, timer, 0);

    glutMainLoop();

    return 0;
    //setup();
    //onDrawFrame();
}
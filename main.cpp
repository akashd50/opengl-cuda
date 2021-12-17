#pragma once
#include <iostream>
#include <math.h>
#include "glm/glm.hpp"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
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
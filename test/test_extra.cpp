#include <iostream>
#include <math.h>
#include "glm/glm.hpp"
#include "GL/glew.h"
#include "GL/freeglut.h"

int main()
{
    glm::vec3 a;
    a.x = 6;
    glewInit();
    std::cout << "MAIN: " << a.x << std::endl;
    return 0;
}
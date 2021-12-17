#pragma once
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "FileReader.h"
#include "ShaderConst.h"

class Shader {
private:
    GLuint vertexShader;
    GLuint fragmentShader;
    GLuint shaderProgram;
public:
    Shader();

    Shader(std::string vShader, std::string fShader);

    void generateVShader(std::string filename);

    void generateFShader(std::string filename);

    GLuint getProgram();

    void useProgram();

    void setUniformMatrix4fv(std::string fname, glm::f32* matrix);

    void setUniform3f(std::string fname, glm::vec3 vec);

    void setUniform4fv(std::string fname, glm::vec4 vec);

    void setUniform1f(std::string fname, float f);

    void setUniformBool(std::string fname, bool f);

    void setVertexAttrib3f(std::string fname, glm::vec3 vec);

    void setUniform1i(std::string fname, int i);

    void setTextureUnit2D(int texUnit, unsigned int texture);

    void setTextureUnit2D(std::string textureUnit, int texUnit, unsigned int texture);

    void setTextureUnit3D(std::string textureUnit, int texUnit, unsigned int texture);
};


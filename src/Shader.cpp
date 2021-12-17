#include "headers/Shader.h"
#include "headers/ShaderConst.h"

Shader::Shader() {

}

Shader::Shader(std::string vShader, std::string fShader) {
    generateVShader(vShader);
    generateFShader(fShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int  success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        exit(0);
    }
    //delete shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Shader::generateVShader(std::string filename) {
    vertexShader = glCreateShader(GL_VERTEX_SHADER);

    std::string shaderCode = FileReader::readTextFile(filename);
    const char* shaderSource = shaderCode.c_str();

    glShaderSource(vertexShader, 1, &shaderSource, NULL);
    glCompileShader(vertexShader);

    GLint success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::VERTEX SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        exit(0);
    }
}

void Shader::generateFShader(std::string filename) {
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    std::string shaderCode = FileReader::readTextFile(filename);
    const char* shaderSource = shaderCode.c_str();

    glShaderSource(fragmentShader, 1, &shaderSource, NULL);
    glCompileShader(fragmentShader);

    GLint success1;
    char infoLog1[512];
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success1);
    if (!success1) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog1);
        std::cout << "ERROR::FRAGMENT SHADER::COMPILATION_FAILED" << infoLog1 << std::endl;
        exit(0);
    }
}

GLuint Shader::getProgram() {
    return shaderProgram;
}

void Shader::useProgram() {
    glUseProgram(shaderProgram);
}

void Shader::setUniformMatrix4fv(std::string fname, glm::f32* matrix) {
    int location = glGetUniformLocation(shaderProgram, fname.c_str());
    glUniformMatrix4fv(location, 1, GL_FALSE, matrix);
}

void Shader::setUniform3f(std::string fname, glm::vec3 vec) {
    int location = glGetUniformLocation(shaderProgram, fname.c_str());
    glUniform3f(location, vec.x, vec.y, vec.z);
}

void Shader::setUniform4fv(std::string fname, glm::vec4 vec) {
    int location = glGetUniformLocation(shaderProgram, fname.c_str());
    glUniform4fv(location, 1, glm::value_ptr(vec));
}

void Shader::setUniform1f(std::string fname, float f) {
    int location = glGetUniformLocation(shaderProgram, fname.c_str());
    glUniform1f(location, f);
}

void Shader::setUniformBool(std::string fname, bool f) {
    int location = glGetUniformLocation(shaderProgram, fname.c_str());
    glUniform1f(location, f);
}

void Shader::setVertexAttrib3f(std::string fname, glm::vec3 vec) {
    int location = glGetAttribLocation(shaderProgram, fname.c_str());
    glVertexAttrib3f(location, vec.x, vec.y, vec.z);
}

void Shader::setUniform1i(std::string fname, int i) {
    int location = glGetUniformLocation(shaderProgram, fname.c_str());
    glUniform1i(location, i);
}

void Shader::setTextureUnit2D(int texUnit, unsigned int texture) {
    glActiveTexture(GL_TEXTURE0 + texUnit);
    glBindTexture(GL_TEXTURE_2D, texture);
}

void Shader::setTextureUnit2D(std::string textureUnit, int texUnit, unsigned int texture) {
    setUniform1i(textureUnit.c_str(), texUnit);
    glActiveTexture(GL_TEXTURE0 + texUnit);
    glBindTexture(GL_TEXTURE_2D, texture);
}

void Shader::setTextureUnit3D(std::string textureUnit, int texUnit, unsigned int texture) {
    setUniform1i(textureUnit.c_str(), texUnit);
    glActiveTexture(GL_TEXTURE0 + texUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
}

//void setSceneProperties(JSONLoader* loader) {
//    setTextureUnit2D(IN_OBJECT_MAPPING_TEXTURE, 1, loader->objectDataMappingTexture);
//    for (int i = 0; i < loader->num_object_mappings; i++) {
//        setUniform1i(IN_OBJECT_MAPPING_INDICES + "[" + std::to_string(i) + "]", loader->objectMappingIndices.at(i));
//    }
//    setUniform1i(IN_NUM_OBJECT_MAPPINGS, loader->num_object_mappings);

//    std::vector<Light*> lights = loader->sceneObject->lights;
//    setUniform1i(IN_NUM_LIGHTS, lights.size());
//    for (int i = 0; i < lights.size(); i++) {
//        Light* light = lights.at(i);
//        std::string indexStr = "[" + std::to_string(i) + "]";
//        setUniform1i(IN_LIGHTS + indexStr + ".type", light->type);
//        setUniform3f(IN_LIGHTS + indexStr + ".color", light->color);

//        if (light->type == POINT_LIGHT) {
//            setUniform3f(IN_LIGHTS + indexStr + ".position", ((PointLight*)light)->position);
//        }
//        
//        if (light->type == SPOT_LIGHT) {
//            setUniform3f(IN_LIGHTS + indexStr + ".position", ((SpotLight*)light)->position);
//            setUniform3f(IN_LIGHTS + indexStr + ".direction", ((SpotLight*)light)->direction);
//            setUniform1f(IN_LIGHTS + indexStr + ".cutoff", ((SpotLight*)light)->cutoff);
//        }

//        if (light->type == DIR_LIGHT) {
//            setUniform3f(IN_LIGHTS + indexStr + ".direction", ((DirectionalLight*)light)->direction);
//        }
//    }

//    // set camera properties
//    setUniform3f(IN_CAMERA + "." + BACKGROUND, loader->sceneObject->camera.background);
//    setUniform1f(IN_CAMERA + "." + REFRACTION, loader->sceneObject->camera.refraction);
//}
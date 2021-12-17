#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "headers/ShaderConst.h"
#include "headers/Quad.h"

Quad::Quad() : Drawable() {}

void Quad::build(Shader* shader) {
    Drawable::build(shader);

    float scale = 1.0;

    glm::vec4 vertices[] = {glm::vec4(1.0f * scale,  1.0f * scale, 0.0f, 1.0f),
                            glm::vec4(-1.0f * scale,  1.0f * scale, 0.0f, 1.0f),
                            glm::vec4(-1.0f * scale, -1.0f * scale, 0.0f, 1.0f),
                            glm::vec4(1.0f * scale,  1.0f * scale, 0.0f, 1.0f),
                            glm::vec4(-1.0f * scale, -1.0f * scale, 0.0f, 1.0f),
                            glm::vec4(1.0f * scale, -1.0f * scale, 0.0f, 1.0f)};

    glm::vec2 uvs[] = { glm::vec2(1.0f, 1.0f),
                        glm::vec2(0.0f, 1.0f),
                        glm::vec2(0.0f, 0.0f),
                        glm::vec2(1.0f, 1.0f),
                        glm::vec2(0.0f, 0.0f),
                        glm::vec2(1.0f, 0.0f) };


    // Quad vertices and uvs - not using the element array buffer

    glGenVertexArrays(1, &vertexArrayObject);
    glBindVertexArray(vertexArrayObject);

    glGenBuffers(1, &vertexArrayBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    GLuint vPosition = glGetAttribLocation(shader->getProgram(), "vPosition");
    glEnableVertexAttribArray(vPosition);
    glVertexAttribPointer(vPosition, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glGenBuffers(1, &uvArrayBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvArrayBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
    GLuint vTextureCoords = glGetAttribLocation(shader->getProgram(), "vTextureCoords");
    glEnableVertexAttribArray(vTextureCoords);
    glVertexAttribPointer(vTextureCoords, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);
}

void Quad::onDrawFrame() {
    glBindVertexArray(vertexArrayObject); // bind vertex array

    // set model matrix and texture
    shader->setUniformMatrix4fv(IN_MODEL, glm::value_ptr(modelMatrix));
    if (getTexture() != NULL) {
        shader->setTextureUnit2D(IN_TEXTURE, 0, getTexture()->getTextureId());
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindVertexArray(0); // unbind vertex array
}
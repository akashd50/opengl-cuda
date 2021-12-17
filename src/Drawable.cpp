#include "headers/Drawable.h"

int Drawable::OBJECTS_ID = 0;

void Drawable::onDrawFrame() { }

GLuint Drawable::getVAO() {
    return this->vertexArrayObject;
}

GLuint Drawable::getVBO() {
    return this->vertexArrayBuffer;
}

Drawable::Drawable() {
    this->ID = OBJECTS_ID++;
    this->translation = new glm::vec3(0.0);
    this->rotation = new glm::vec3(0.0);
    this->scale = new glm::vec3(1.0);
    this->parent = NULL;
    this->colours = NULL;
    this->usesSingleColor = true;
}

void Drawable::build(Shader* shader) {
    this->shader = shader;
}

glm::mat4 Drawable::applyTransformations() {
    /*
     * Apply Transformations
     * if this drawable has a parent - applies the parent transformations first
     * then, translate - rotate - scale
     */
    if (this->parent != NULL) {
        modelMatrix = parent->applyTransformations();
    } else {
        modelMatrix = glm::mat4();
    }

    modelMatrix = glm::translate(modelMatrix, *translation);
    modelMatrix = glm::rotate(modelMatrix, glm::radians(rotation->x), glm::vec3(1.0f, 0.0f, 0.0f));
    modelMatrix = glm::rotate(modelMatrix, glm::radians(rotation->y), glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix = glm::rotate(modelMatrix, glm::radians(rotation->z), glm::vec3(0.0f, 0.0f, 1.0f));
    modelMatrix = glm::scale(modelMatrix, *scale);
    return modelMatrix;
}

/*-------------------------------------- Helper Functions - Getters and Setters ----------------------------------------*/
void Drawable::setTexture(Texture* texture) {
    this->texture = texture;
}

Texture* Drawable::getTexture() {
    return this->texture;
}

Shader* Drawable::getShader() {
    return this->shader;
}

void Drawable::setColours(int len, glm::vec4* colours) {
    this->numColours = len;
    this->colours = colours;
}

void Drawable::setUsesSingleColor(bool val) {
    this->usesSingleColor = val;
}

void Drawable::setParent(Drawable* parent) {
    this->parent = parent;
}

Drawable* Drawable::getParent() {
    return this->parent;
}

void Drawable::translateTo(glm::vec3 position) {
    this->translation->x = position.x;
    this->translation->y = position.y;
    this->translation->z = position.z;
}
void Drawable::translateBy(glm::vec3 translation) {
    this->translation->x += translation.x;
    this->translation->y += translation.y;
    this->translation->z += translation.z;
}

void Drawable::rotateTo(glm::vec3 rotation) {
    this->rotation->x = rotation.x;
    this->rotation->y = rotation.y;
    this->rotation->z = rotation.z;
    capRotation();
}

void Drawable::rotateBy(glm::vec3 rotation) {
    this->rotation->x += rotation.x;
    this->rotation->y += rotation.y;
    this->rotation->z += rotation.z;
    capRotation();
}

void Drawable::scaleTo(glm::vec3 scale) {
    this->scale->x = scale.x;
    this->scale->y = scale.y;
    this->scale->z = scale.z;
}

void Drawable::scaleBy(glm::vec3 scale) {
    this->scale->x += scale.x;
    this->scale->y += scale.y;
    this->scale->z += scale.z;
}

void Drawable::capRotation() {
    float rotX = (*this->rotation).x;
    (*this->rotation).x = rotX > 360? rotX - 360 : rotX;
    (*this->rotation).x = rotX < -360 ? rotX + 360 : rotX;

    float rotY = (*this->rotation).y;
    (*this->rotation).y = rotY > 360 ? rotY - 360 : rotY;
    (*this->rotation).y = rotY < -360 ? rotY + 360 : rotY;

    float rotZ = (*this->rotation).z;
    (*this->rotation).z = rotZ > 360 ? rotZ - 360 : rotZ;
    (*this->rotation).z = rotZ < -360 ? rotZ + 360 : rotZ;
}

void Drawable::translateTo(float x, float y, float z) {
    translateTo(glm::vec3(x, y, z));
}

void Drawable::translateBy(float x, float y, float z) {
    translateBy(glm::vec3(x, y, z));
}

void Drawable::rotateTo(float x, float y, float z) {
    rotateTo(glm::vec3(x, y, z));
}

void Drawable::rotateBy(float x, float y, float z) {
    rotateBy(glm::vec3(x, y, z));
}

void Drawable::scaleTo(float x, float y, float z) {
    scaleTo(glm::vec3(x, y, z));
}

void Drawable::scaleBy(float x, float y, float z) {
    scaleBy(glm::vec3(x, y, z));
}

glm::vec3* Drawable::getTranslation() {
    return this->translation;
}

glm::vec3* Drawable::getRotation() {
    return this->rotation;
}

glm::vec3* Drawable::getScale() {
    return this->scale;
}

int Drawable::getID() {
    return this->ID;
}
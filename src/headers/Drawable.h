#pragma once
#include "Shader.h"
#include "Texture.h"

class Drawable {
protected:
	int ID, numColours;
	Shader* shader;
	glm::vec3* translation, * rotation, * scale;
	glm::mat4 modelMatrix;
	Drawable* parent;
	glm::vec4* colours;
	bool usesSingleColor;
	GLuint vertexArrayObject, vertexArrayBuffer, colorArrayBuffer, uvArrayBuffer, elementArrayBuffer;
	Texture* texture;
public:
	static int OBJECTS_ID;

	Drawable();
	void build(Shader* shader);
    void onDrawFrame();
    glm::mat4 applyTransformations();

	/*-------------------------------------- Helper Functions - Getters and Setters ----------------------------------------*/
	void setTexture(Texture* texture);
	Texture* getTexture();

	Shader* getShader();

	void setColours(int len, glm::vec4* colours);
	void setUsesSingleColor(bool val);

	void setParent(Drawable* parent);
	Drawable* getParent();

	void translateTo(glm::vec3 position);
	void translateBy(glm::vec3 translation);

	void rotateTo(glm::vec3 rotation);
	void rotateBy(glm::vec3 rotation);

	void scaleTo(glm::vec3 scale);
	void scaleBy(glm::vec3 scale);

	void capRotation();

	void translateTo(float x, float y, float z);
	void translateBy(float x, float y, float z);

	void rotateTo(float x, float y, float z);
	void rotateBy(float x, float y, float z);

	void scaleTo(float x, float y, float z);
	void scaleBy(float x, float y, float z);

	glm::vec3* getTranslation();
	glm::vec3* getRotation();
	glm::vec3* getScale();

	int getID();
    GLuint getVAO();
    GLuint getVBO();
};

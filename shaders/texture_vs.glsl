#version 150
in vec4 vPosition;
in vec2 vTextureCoords;

out vec2 textureCoords;

void main() {
	textureCoords = vTextureCoords;
	gl_Position = vPosition;
}

#version 150

uniform sampler2D in_texture;
in vec2 textureCoords;

out vec4 out_colour;

void main() {
    vec4 textureCol = texture(in_texture, textureCoords);
    out_colour = textureCol;
}

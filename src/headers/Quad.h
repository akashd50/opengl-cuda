#pragma once
#include "Drawable.h"

class Quad: public Drawable {
public:
	Quad();
    void build(Shader* shader);
    void onDrawFrame();
};


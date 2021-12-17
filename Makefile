CFLAGS=-std=c++17

GLM_PATH=C:/dev_libs/glm-0.9.9.8/glm
GLEW_PATH=C:/dev_libs/glew
FREEGLUT_PATH=C:/dev_libs/freeglut

SRC=$(wildcard *.cpp)
INCLUDES=-I$(GLM_PATH) -I$(FREEGLUT_PATH)/include -I$(GLEW_PATH)/include
FRAMEWORKS_2=-lopengl32 -lglu32

LIBDIRS=-L$(GLEW_PATH)/lib -L$(FREEGLUT_PATH)/lib
LIBS=-lglew32 -lfreeglut

all: MainCuda MainOpenGL
	nvcc $(CFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(FRAMEWORKS_2) MainCuda.obj MainOpenGL.obj -o Main

MainCuda: main_cu.cu
	nvcc $(CFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(FRAMEWORKS_2) -c main_cu.cu -o MainCuda

MainOpenGL: MainOpenGL.cpp
	nvcc $(CFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(FRAMEWORKS_2) -c MainOpenGL.cpp -o MainOpenGL

test:
	@echo Test
	@echo src: %$(SRC)%

clean:
	del -f Main.exe
	del -f Main.exp
	del -f Main.lib
	del -f Main.pdb
	del -f MainOpenGL.pdb
	del -f MainOpenGL.obj
	del -f MainCuda.pdb
	del -f MainCuda.obj
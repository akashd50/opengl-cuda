CFLAGS=-std=c++17 -g -DDEBUG

GLM_PATH=C:/dev_libs/glm-0.9.9.8/glm
GLEW_PATH=C:/dev_libs/glew
FREEGLUT_PATH=C:/dev_libs/freeglut

SRC=$(wildcard *.c)
INCLUDES=-I$(GLM_PATH) -I$(FREEGLUT_PATH)/include -I$(GLEW_PATH)/include
FRAMEWORKS=-framework OpenGL -framework GLUT
FRAMEWORKS_2=-lopengl32 -lglu32

LIBDIRS=-L$(GLEW_PATH)/lib -L$(FREEGLUT_PATH)/lib
LIBS=-lglew32 -lfreeglut

CudaMain: main_cu.cu
	nvcc $(CFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) $(FRAMEWORKS_2) main_cu.cu -o CudaMain

all: CudaMain
	CudaMain

clean:
	del -f CudaMain.exe
	del -f CudaMain.exp
	del -f CudaMain.lib
	del -f CudaMain.pdb
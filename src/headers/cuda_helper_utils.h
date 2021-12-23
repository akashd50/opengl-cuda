#pragma once
#include <vector_types.h>
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "cuda_classes.h"

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------Helper--Functions-------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line);

//template <class T>
//T* cudaWrite(T* data, int len);
//template <class T>
//T* cudaRead(T* data, int len);


Bounds* getNewBounds(std::vector<CudaTriangle*>* triangles);
bool isFloat3InBounds(float3 point, Bounds* bounds);
bool isTriangleInBounds(CudaTriangle* triangle, Bounds* bounds);
BVHBinaryNode* createTreeHelper(std::vector<CudaTriangle*>* localTriangles, BVHBinaryNode* node);
BVHBinaryNode* createHostTreeHelper(std::vector<CudaTriangle*>* localTriangles, BVHBinaryNode* node);
void cleanCudaScene(CudaScene* scene);

CudaScene* allocateCudaScene(CudaScene* scene);
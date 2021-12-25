#pragma once
#include <vector_types.h>
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <string>
#include "cuda_classes.h"

//----------------------------------------------------------------------------------------------------------------------
//----------------------------------------------Helper--Functions-------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------
#define check(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, const std::string& file, int line);

//template <class T>
//T* cudaWrite(T* data, int len);
//template <class T>
//T* cudaRead(T* data, int len);

CudaScene* allocateCudaScene(CudaScene* scene);
void cleanCudaScene(CudaScene* scene);
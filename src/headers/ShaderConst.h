#pragma once
#include <iostream>
#include <string>
#include <cuda_runtime.h>

const static std::string IN_MODEL = "model";
const static std::string IN_TEXTURE = "in_texture";
const static std::string IN_RAY_TEXTURE = "in_ray_texture";
const static std::string IN_RAND_TEXTURE = "in_rand_texture";
const static std::string IN_OBJECT_MAPPING_TEXTURE = "in_object_mapping_texture";
const static std::string IN_OBJECT_MAPPING_INDICES = "in_object_mapping_indices";
const static std::string IN_NUM_OBJECT_MAPPINGS = "in_num_object_mappings";
const static std::string IN_LIGHTS = "in_lights";
const static std::string IN_NUM_LIGHTS = "in_num_lights";
const static std::string IN_SX_OFFSET = "in_sx_offset";
const static std::string IN_SY_OFFSET = "in_sy_offset";
const static std::string IN_CAMERA_VIEW = "in_camera_view";
const static std::string IN_CAMERA = "in_camera";
const static std::string BACKGROUND = "background";
const static std::string REFRACTION = "refraction";

const static std::string IN_LIGHTING_TEX_MAIN = "lighting_texture_main";
const static std::string IN_LIGHTING_TEX_I = "lighting_texture_i";
const static std::string IN_LIGHTING_TEX_II= "lighting_texture_ii";
const static std::string IN_LIGHTING_TEX_III = "lighting_texture_iii";
const static std::string IN_LIGHTING_TEX_IV = "lighting_texture_iv";

__device__ const int SPHERE = 1;
__device__ const int MESH = 2;
__device__ const int TRIANGLE = 3;
__device__ const int PLANE = 4;
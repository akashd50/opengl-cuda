typedef struct Material {
    float3 ambient, diffuse, specular, reflective, transmissive;
    float refraction, roughness, shininess;
} Material;

typedef struct __attribute__((packed)) Sphere {
    int type;
    int material_id;
    float3 position;
    float radius;
} Sphere;

__kernel void vbo_update(__global float4* vbo) {
    float4 coord = vbo[get_global_id(0)];
    vbo[get_global_id(0)] = (float4)(coord.x + 0.05, coord.y, coord.z, coord.w);
}

float3 cast_ray(int x, int y, int width, int height) {
    float d = 1.0;
    float fov = 60.0;
    float aspect_ratio = ((float)width) / ((float)height);
    float h = d * (float)tan((3.1415 * fov) / 180.0 / 2.0);
    float w = h * aspect_ratio;

    float top = h;
    float bottom = -h;
    float left = -w;
    float right = w;

    float u = left + (right - left) * (x) / ((float)width);
    float v = bottom + (top - bottom) * (((float)height) - y) / ((float)height);

    return (float3)(u, v, -d);
}

const float MIN_T = -9999.0;
const float HIT_T_OFFSET = 0.01;

float check_hit_on_sphere(float3 eye, float3 ray, float3 center, float radius) {
    float3 center_2_eye = eye - center;
    float ray_dot_ray = dot(ray, ray);
    float discriminant = pow(dot(ray, center_2_eye), 2) - ray_dot_ray * (dot(center_2_eye, center_2_eye) - pow(radius, 2));

    if (discriminant > 0) {
        discriminant = sqrt(discriminant);
        float init = -dot(ray, center_2_eye);
        float t1 = (init + discriminant) / ray_dot_ray;
        float t2 = (init - discriminant) / ray_dot_ray;

        float mint = min(t1, t2);
        if (mint < HIT_T_OFFSET) {
            return max(t1, t2);
        }
        return mint;
    }
    else if (discriminant == 0) {
        float init = -dot(ray, center_2_eye);
        float t1 = init / ray_dot_ray;
        return t1;
    }
    return MIN_T;
}

__kernel void mod_rgb_texture(__write_only image2d_t tex, global int* dim, global Sphere* sp) {
    // const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    int2 index = (int2) (get_global_id(0), get_global_id(1));

    //float4 pixel = read_imagef(image, smp, index);
    //write_imagef(tex, index, (float4)(0.0, 1.0, 0.0, 1.0));
    //printf("Pos: %f %f %f\n", sp->position.x, sp->position.y, sp->position.z);
    float3 pos = (float3)(0.0, 0.0, -5.0);
    float3 eye = (float3)(0.0, 0.0, 0.0);
    float3 ray = cast_ray(index.x, index.y, dim[0], dim[1]) - eye;
    float sphereHit = check_hit_on_sphere(eye, ray, pos, sp->radius);

    if (sphereHit >= 0 && sphereHit != MIN_T) {
        write_imagef(tex, index, (float4)(1.0, 0.0, 0.0, 1.0));
    }
    else {
        write_imagef(tex, index, (float4)(0.0, 0.0, 0.0, 1.0));
    }
    
    //write_imagef(tex, index, (float4)(ray, 1.0));

    //float4 coord = tex[get_global_id(0)];
    //tex[get_global_id(0)] = (float4)(1.0, 0.0, 0.0, 1.0);
}
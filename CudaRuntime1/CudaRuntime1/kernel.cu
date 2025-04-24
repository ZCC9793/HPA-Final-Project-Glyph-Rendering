#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <cstring>
#include <ctime> 

#include "letters.h"
#include "1.h"
#include "2.h"
#include "3.h"
#include "4.h"

#define GLYPH_COUNT 26
#define BLOCK_SIZE 32
#define GLYPH_MAX_SIZE 256
#define IMAGE_WIDTH 1200
#define IMAGE_HEIGHT 1400
#define CHANNELS 4

using namespace std;

struct Color { unsigned char r, g, b; };
struct Transform { Point scale_x, scale_y, offset; };
struct ImageData { unsigned char* pixels; int width, height; };
struct Glyph { LineSegment* lines; int line_count; };
struct BoundingBox { Point low, high; };

void try_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

__host__ __device__ Point apply_transform(Point p, Transform t) {
    return {
        p.x * t.scale_x.x + p.y * t.scale_x.y + t.offset.x,
        p.x * t.scale_y.x + p.y * t.scale_y.y + t.offset.y
    };
}

__host__ __device__ float sqr(float x) { return x * x; }
__host__ __device__ float dist2(Point v, Point w) { return sqr(v.x - w.x) + sqr(v.y - w.y); }
__host__ __device__ float clamp(float x, float low, float high) {
    return fminf(fmaxf(x, low), high);
}

__host__ __device__ float dist_to_segment_squared(Point p, Point p1, Point p2) {
    float l2 = dist2(p1, p2);
    if (l2 == 0.0f) return dist2(p, p1);
    float t = clamp(((p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y)) / l2, 0.0f, 1.0f);
    return dist2(p, { p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y) });
}

__host__ __device__ bool is_point_inside(Point p, LineSegment* lines, int count, Transform t) {
    int crossings = 0;
    for (int i = 0; i < count; ++i) {
        Point s = apply_transform(lines[i].start, t);
        Point e = apply_transform(lines[i].end, t);
        if ((s.y > p.y) != (e.y > p.y)) {
            float atX = (e.x - s.x) * (p.y - s.y) / (e.y - s.y + 1e-6f) + s.x;
            if (p.x < atX) ++crossings;
        }
    }
    return (crossings % 2) != 0;
}

__host__ __device__ void setPixel(ImageData img, Point pos, Color color) {
    int x = (int)pos.x;
    int y = (int)pos.y;
    if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
    int i = (y * img.width + x) * 4;
    img.pixels[i + 0] = color.r;
    img.pixels[i + 1] = color.g;
    img.pixels[i + 2] = color.b;
    img.pixels[i + 3] = 255;
}

__host__ __device__ Point bounding_box_size(BoundingBox b) {
    return { b.high.x - b.low.x, b.high.y - b.low.y };
}

BoundingBox find_bounding_box(Transform t, float size) {
    Point corners[4] = {
        apply_transform({0, 0}, t),
        apply_transform({0, size}, t),
        apply_transform({size, 0}, t),
        apply_transform({size, size}, t)
    };
    BoundingBox box = { corners[0], corners[0] };
    for (int i = 1; i < 4; ++i) {
        box.low.x = fminf(box.low.x, corners[i].x);
        box.low.y = fminf(box.low.y, corners[i].y);
        box.high.x = fmaxf(box.high.x, corners[i].x);
        box.high.y = fmaxf(box.high.y, corners[i].y);
    }
    return box;
}

__global__ void draw_glyph_fill_and_stroke_gpu(ImageData img, Glyph glyph, Transform t, Color color, float stroke_weight, BoundingBox box) {
    extern __shared__ LineSegment shared_lines[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < glyph.line_count) shared_lines[tid] = glyph.lines[tid];
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x + (int)box.low.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y + (int)box.low.y;
    if (x > box.high.x || y > box.high.y) return;

    Point pos = { (float)x, (float)y };
    bool inside = is_point_inside(pos, shared_lines, glyph.line_count, t);

    float min_dist = 1e9f;
    for (int i = 0; i < glyph.line_count; ++i) {
        Point s = apply_transform(shared_lines[i].start, t);
        Point e = apply_transform(shared_lines[i].end, t);
        min_dist = fminf(min_dist, dist_to_segment_squared(pos, s, e));
    }

    if (inside || min_dist < stroke_weight) setPixel(img, pos, color);
    // else setPixel(img, pos, { 100, 0, 0 });
}

bool launch_draw_glyph_fill_and_stroke_gpu(ImageData& img, const char* text, Color color, float stroke_weight) {
    for (int i = 0; text[i]; i++) {
        char ch = text[i];
        if (ch < 'A' || ch > 'Z') continue;
        int index = ch - 'A';

        LineSegment* host_lines = letters_basic[index];
        int line_count = letter_line_counts[index];

        LineSegment* device_lines;
        cudaError_t err = cudaMalloc(&device_lines, sizeof(LineSegment) * line_count);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        cudaMemcpy(device_lines, host_lines, sizeof(LineSegment) * line_count, cudaMemcpyHostToDevice);

        Glyph device_glyph = { device_lines, line_count };

        Transform t = {
            { 0.4f, 0.0f },
            { 0.0f, 0.4f },
            { 50.0f + i * 150.0f, 100.0f }
        };

        BoundingBox box = { { 0, 0 }, { (float)IMAGE_WIDTH, (float)IMAGE_HEIGHT } };
        Point size = bounding_box_size(box);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((int)ceil(size.x / BLOCK_SIZE), (int)ceil(size.y / BLOCK_SIZE));
        size_t shared_mem = sizeof(LineSegment) * line_count;

        draw_glyph_fill_and_stroke_gpu << <grid, block, shared_mem >> > (img, device_glyph, t, color, stroke_weight, box);
        cudaDeviceSynchronize();

        cudaFree(device_lines);
    }
    return cudaGetLastError() == cudaSuccess;
}

void draw_glyph_fill_and_stroke_cpu(ImageData& img, const char* text, Color color, float stroke_weight) {
    for (int i = 0; text[i]; i++) {
        char ch = text[i];
        if (ch < 'A' || ch > 'Z') continue;
        int index = ch - 'A';

        LineSegment* lines = letters_basic[index];
        int line_count = letter_line_counts[index];

        Transform t = {
            { 1.0f, 0.0f },
            { 0.0f, 1.0f },
            { 50.0f + i * 150.0f, 300.0f }
        };
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                Point pos = { (float)x, (float)y };
                bool inside = is_point_inside(pos, lines, line_count, t);
                float min_dist = 1e9f;
                for (int l = 0; l < line_count; ++l) {
                    Point s = apply_transform(lines[l].start, t);
                    Point e = apply_transform(lines[l].end, t);
                    min_dist = fminf(min_dist, dist_to_segment_squared(pos, s, e));
                }
                if (inside || min_dist < stroke_weight) setPixel(img, pos, color);
            }
        }
    }
}


int divide_ceil(int a, int b) { return (a + b - 1) / b; }

LineSegment* font_host[GLYPH_COUNT] = {
    lines_for_A, lines_for_B, lines_for_C, lines_for_D, lines_for_E, lines_for_F,
    lines_for_G, lines_for_H, lines_for_I, lines_for_J, lines_for_K, lines_for_L,
    lines_for_M, lines_for_N, lines_for_O, lines_for_P, lines_for_Q, lines_for_R,
    lines_for_S, lines_for_T, lines_for_U, lines_for_V, lines_for_W, lines_for_X,
    lines_for_Y, lines_for_Z
};

template<typename T>
T* copy_to_gpu(T* data, int count) {
    T* gpu_ptr;
    try_cuda(cudaMalloc(&gpu_ptr, count * sizeof(T)));
    try_cuda(cudaMemcpy(gpu_ptr, data, count * sizeof(T), cudaMemcpyHostToDevice));
    return gpu_ptr;
}

//void draw_text(ImageData img, LineSegment** font, int* line_counts, const char* text, Transform base_t, float spacing, float stroke_w, Color color) {
//    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
//
//    for (int i = 0; text[i]; ++i) {
//        int index = text[i] - 'A';
//        if (index < 0 || index >= GLYPH_COUNT) continue;
//
//        LineSegment* lines_gpu = copy_to_gpu(font[index], line_counts[index]);
//        Glyph glyph = { lines_gpu, line_counts[index] };
//
//        Transform t = base_t;
//        t.offset.x += spacing * i;
//
//        BoundingBox bbox = find_bounding_box(t, GLYPH_MAX_SIZE);
//        Point size = bounding_box_size(bbox);
//        dim3 grid(divide_ceil((int)size.x, BLOCK_SIZE), divide_ceil((int)size.y, BLOCK_SIZE));
//
//        size_t shared_mem_size = line_counts[index] * sizeof(LineSegment);
//        draw_glyph_fill_and_stroke_gpu << <grid, block, shared_mem_size >> > (img, glyph, t, color, stroke_w, bbox);
//        try_cuda(cudaFree(lines_gpu));
//    }
//    try_cuda(cudaDeviceSynchronize());
//}

void write_image(const string& filename, unsigned char* pixels, int w, int h) {
    stbi_write_png(filename.c_str(), w, h, 4, pixels, w * 4);
}

BoundingBox get_bounding_box(Transform t, float size = 256.0f) {
    Point corners[] = {
        apply_transform({0, 0}, t),
        apply_transform({size, 0}, t),
        apply_transform({0, size}, t),
        apply_transform({size, size}, t)
    };
    Point low = corners[0], high = corners[0];
    for (int i = 1; i < 4; ++i) {
        low.x = fminf(low.x, corners[i].x);
        low.y = fminf(low.y, corners[i].y);
        high.x = fmaxf(high.x, corners[i].x);
        high.y = fmaxf(high.y, corners[i].y);
    }
    return { low, high };
}


int main() {
    const char* text = "ZACHARY";  // Manually set your text string here
    bool error;
    clock_t start, end;
    float time_cpu, time_gpu;
    int iters_host = 1;
    int iters_device = 3;
    unsigned char* pixels = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];
    memset(pixels, 0, IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS);

    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        pixels[i * 4 + 3] = 255; // fully opaque alpha
    }

    ImageData image;
    cudaMalloc(&image.pixels, IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS);
    cudaMemcpy(image.pixels, pixels, IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS, cudaMemcpyHostToDevice);
    image.width = IMAGE_WIDTH;
    image.height = IMAGE_HEIGHT;

    ImageData image_cpu;
    int image_size = CHANNELS * IMAGE_WIDTH * IMAGE_HEIGHT;
    image_cpu.pixels = (unsigned char*)malloc(image_size);
    if (image_cpu.pixels == nullptr) {
        cout << "Failed to malloc" << endl;
        abort();
    }

    memset(image_cpu.pixels, 0, image_size);


    image_cpu.width = IMAGE_WIDTH;
    image_cpu.height = IMAGE_HEIGHT;

    Color color = { 255, 255, 255 };
    float stroke_weight = 4.0f;

    error = launch_draw_glyph_fill_and_stroke_gpu(image, text, color, stroke_weight);
    if (!error) {
        printf("Error: Error Launching GPU Kernel\n");
        return 1;
    }

    start = clock();
    for (int i = 0; i < iters_device; i++) {
        launch_draw_glyph_fill_and_stroke_gpu(image, text, color, stroke_weight);
    }
    end = clock();

    time_gpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iters_device;
  
    cout << "GPU Implementation took " << time_gpu << "ms" << endl;

    start = clock();
    for (int i = 0; i < iters_host; i++) {
        draw_glyph_fill_and_stroke_cpu(image_cpu, text, { 0, 255, 0 }, stroke_weight);

    }
    end = clock();

    time_cpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / iters_host;

    cout << "CPU Implementation took " << time_cpu << "ms" << endl;

    cudaMemcpy(pixels, image.pixels, IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS, cudaMemcpyDeviceToHost);
    stbi_write_png("glyph_output.png", IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, pixels, IMAGE_WIDTH * CHANNELS);

    cudaFree(image.pixels);
    delete[] pixels;
    return 0;
}

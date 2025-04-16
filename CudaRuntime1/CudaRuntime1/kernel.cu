
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <cmath>

#include "Letters.h"

using namespace std;

typedef unsigned char ImageData;

void try_cuda(cudaError_t error) {
	if (error == cudaSuccess) return;

	const char* error_string = cudaGetErrorString(error);
	cout << "CUDA Error: " << error_string << endl;
	abort();
}



int width = 1920;
int height = 1080;
int channels = 4;


struct Color {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

__global__ void addKernel(int* c, const int* a, const int* b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__host__ __device__  float dot_product(Point p1, Point p2) {
	return p1.x * p2.x + p1.y * p2.y;
}

__host__ __device__  float cross_product(Point p1, Point p2) {
	return p1.x * p2.y - p1.y * p2.x;
}

__host__ __device__  Point subtract(Point p1, Point p2) {
	return { p1.x - p2.x, p1.y - p2.y };
}

__host__ __device__  Point add(Point p1, Point p2) {
	return { p1.x + p2.x, p1.y + p2.y };
}

__host__ __device__ float get_ray_to_line_segment_intersection(Point ray_origin, Point ray_direction, Point p1, Point p2) {
	Point ray_local = subtract(ray_origin, p1);
	Point p2_local = subtract(p2, p1);
	Point ray_perp = { -ray_direction.y, ray_direction.x };

	float dot = dot_product(p2_local, ray_perp);
	if (fabs(dot) < 0.001) return -1;

	float t1 = cross_product(p2_local, ray_local) / dot;
	float t2 = dot_product(ray_local, ray_perp) / dot;

	if (t1 < 0.0 || (t2 < 0.0 || t2 > 1.0)) return -1;

	return t1;
}

__host__ __device__ void setPixel(ImageData* image_data, int x, int y, int width, Color color) {
	image_data[(y * width + x) * 4 + 0] = color.r;
	image_data[(y * width + x) * 4 + 1] = color.g;
	image_data[(y * width + x) * 4 + 2] = color.b;
}

__host__ __device__ void draw_glyph_pixel(ImageData* image_data, LineSegment* lines, int line_count, Color color, Point offset, int width, int x, int y) {
	bool inside = false;
	for (int l = 0; l < line_count; l++) {
		LineSegment line = lines[l];
		Point start = add(line.start, offset);
		Point end = add(line.end, offset);

		// skip horizontal edges
		if (start.y == end.y) continue;

		Point ray_origin = { x + 0.5f, y + 0.5f }; // shift to pixel center
		Point ray_dir = { 1.0f, 0.0f };

		float d = get_ray_to_line_segment_intersection(ray_origin, ray_dir, start, end);

		if (d >= 0) inside = !inside;
	}

	if (inside) setPixel(image_data, x, y, width, color);
}

__global__ void draw_glyph_gpu(ImageData* image_data, LineSegment* lines, int line_count, Color color, Point offset, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	draw_glyph_pixel(image_data, lines, line_count, color, offset, width, x, y);
}

void draw_glyph(ImageData* image_data, LineSegment* lines, int line_count, Color color, Point offset, Point(*transform)(Point), int width) {
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			draw_glyph_pixel(image_data, lines, line_count, color, offset, width, x, y);
		}
	}
}

__host__ __device__ float float_max(float f1, float f2) {
	return f1 > f2 ? f1 : f2;
}

__host__ __device__ float float_min(float f1, float f2) {
	return f1 < f2 ? f1 : f2;
}

__host__ __device__ float sqr(float x) { return x * x; }
__host__ __device__ float dist2(Point v, Point w) { return sqr(v.x - w.x) + sqr(v.y - w.y); }
__host__ __device__ float dist_to_segment_squared(Point p, Point p1, Point p2) {
	float l2 = dist2(p1, p2);
	if (l2 == 0) return dist2(p, p1);
	float t = ((p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y)) / l2;
	float t2 = float_max(0, float_min(1, t));
	return dist2(p, { p1.x + t2 * (p2.x - p1.x), p1.y + t2 * (p2.y - p1.y) });
}

__host__ __device__ void draw_glyph_stroke_pixel(ImageData* image_data, LineSegment* lines, int line_count, float weight, Color color, Point offset, int width, int x, int y) {
	float min_dist = numeric_limits<float>::max();

	for (int l = 0; l < line_count; l++) {
		LineSegment line = lines[l];
		Point start = add(line.start, offset);
		Point end = add(line.end, offset);

		min_dist = min(min_dist, dist_to_segment_squared({ (float)x, (float)y }, start, end));
	}
	if (min_dist < weight) setPixel(image_data, x, y, width, color);
}


__host__ __device__ float map_range(float n, float start1, float stop1, float start2, float stop2) {
	return ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2;
};

__host__ __device__ void draw_glyph_stroke_pixel_wack(ImageData* image_data, LineSegment* lines, int line_count, float weight, Color color, Point offset, int width, int x, int y) {
	float min_dist = numeric_limits<float>::max();

	for (int l = 0; l < line_count; l++) {
		LineSegment line = lines[l];
		Point start = add(line.start, offset);
		Point end = add(line.end, offset);

		min_dist = min(min_dist, dist_to_segment_squared({ (float)x, (float)y }, start, end));
	}

	float s = map_range(min_dist, 0, weight * 100, 1, 0);
	s = float_max(s, 0);
	if (s < 0.01) return;
	setPixel(image_data, x, y, width, { (unsigned char)(color.r * s), (unsigned char)(color.g * s), (unsigned char)(color.b * s) });
}

__global__ void draw_glyph_stroke_gpu(ImageData* image_data, LineSegment* lines, int line_count, float weight, Color color, Point offset, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	draw_glyph_stroke_pixel_wack(image_data, lines, line_count, weight, color, offset, width, x, y);
}

Point scale_down(Point p) {
	float s = 0.8;
	return { p.x * s, p.y * s };
}

Point shear(Point p, float magnitude) {
	return { p.x - p.y * magnitude, p.y };
}

Point scale_down_and_italic(Point p) {
	return shear(scale_down(p), 0.2f);
}

int divide_ceil(int dividend, int divisor) {
	return (dividend + divisor - 1) / divisor;
}

#define BLOCK_SIZE 32
#define LETTER_COUNT 26


template<typename T>
T* copy_to_gpu(T* data, int count) {
	T* data_gpu;
	int data_size = count * sizeof(T);
	try_cuda(cudaMalloc(&data_gpu, data_size));
	try_cuda(cudaMemcpy(data_gpu, data, data_size, cudaMemcpyHostToDevice));
	return data_gpu;
}

void draw_text(ImageData* image_data_gpu, int image_data_size, LineSegment** letters, char* text) {
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size(divide_ceil(width, BLOCK_SIZE), divide_ceil(height, BLOCK_SIZE));

	for (int i = 0; text[i]; i++) {
		cout << i << endl;
		char letter_index = text[i] - 'A';
		LineSegment* letter_gpu = letters[letter_index];

		//draw_glyph_gpu << <grid_size, block_size >> > (image_data_gpu, letter_gpu, letter_line_counts[i], { 255, 0, (unsigned char)(i * 20) }, { (float)i * 256, 30 }, width);
		// draw_glyph_stroke_gpu << <grid_size, block_size >> > (image_data_gpu, letter_gpu, letter_line_counts[i], 5, { 0, 255, (unsigned char)(i * 40) }, { (float)i * 256, 30 }, width);
		draw_glyph_stroke_gpu << <grid_size, block_size >> > (image_data_gpu, letter_gpu, letter_line_counts[i], 5, { 255, (unsigned char)(i * 40), (unsigned char)(255 - i * 40) }, { (float)i * 256, 30 }, width);
		try_cuda(cudaGetLastError());
		try_cuda(cudaDeviceSynchronize());
	}
	// should be here - try_cuda(cudaDeviceSynchronize());
}

int main() {
	int image_data_size = sizeof(unsigned char) * channels * width * height;
	ImageData* image_data = (ImageData*)malloc(image_data_size);
	if (image_data == nullptr) {
		cout << "Failed to malloc" << endl;
		return 0;
	}
	memset(image_data, 0, image_data_size);

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			image_data[(r * width + c) * 4 + 3] = 255;
		}
	}


	int line_count = sizeof(lines_for_A) / sizeof(LineSegment);

	// draw_glyph(image_data, lines_for_A, line_count, { 255, 0, 0 }, { 20, 0 }, scale_down_and_italic, width);
	// draw_glyph_stroke(image_data, lines_for_A, line_count, 30, { 0, 0, 255 }, { 20, 0 }, scale_down_and_italic, width);

	ImageData* image_data_gpu = copy_to_gpu(image_data, channels * width * height);

	LineSegment** letters = new LineSegment * [LETTER_COUNT];
	for (int i = 0; i < LETTER_COUNT; i++) {
		LineSegment* letter_gpu = copy_to_gpu(letters_basic[i], letter_line_counts[i]);
		letters[i] = letter_gpu;
	}

	draw_text(image_data_gpu, image_data_size, letters, "ABCDEFG");

	try_cuda(cudaMemcpy(image_data, image_data_gpu, image_data_size, cudaMemcpyDeviceToHost));

	stbi_write_png("test_out.png", width, height, channels, image_data, width * channels);

	cout << "Image written" << endl;

	free(image_data);

	return 0;
}
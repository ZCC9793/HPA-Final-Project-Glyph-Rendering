
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

#define TRY_CUDA(x) if (!check_for_error(x)) return false;
bool check_for_error(cudaError_t error) {
	if (error == cudaSuccess) return true;

	const char* error_string = cudaGetErrorString(error);
	cout << "CUDA Error: " << error_string << endl;

	return false;
}



int width = 3080;
int height = 2160;
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

__host__ __device__ void setPixel(unsigned char* image_data, int x, int y, int width, Color color) {
	image_data[(y * width + x) * 4 + 0] = color.r;
	image_data[(y * width + x) * 4 + 1] = color.g;
	image_data[(y * width + x) * 4 + 2] = color.b;
}

__global__ void draw_glyph_gpu(unsigned char* image_data, LineSegment* lines, int line_count, Color color, Point offset, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	bool inside = false;
	for (int l = 0; l < line_count; l++) {
		LineSegment line = lines[l];
		Point start = add(line.start, offset);
		Point end = add(line.end, offset);

		// ✅ Skip horizontal edges
		if (start.y == end.y) continue;

		Point ray_origin = { x + 0.5f, y + 0.5f };  // shift to pixel center
		Point ray_dir = { 1.0f, 0.0f };

		float d = get_ray_to_line_segment_intersection(ray_origin, ray_dir, start, end);

		if (d >= 0) inside = !inside;
	}

	if (inside) setPixel(image_data, x, y, width, color);
}

void draw_glyph(unsigned char* image_data, LineSegment* lines, int line_count, Color color, Point offset, Point(*transform)(Point), int width) {
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			bool inside = false;
			for (int l = 0; l < line_count; l++) {
				LineSegment line = lines[l];
				Point start = add(transform(line.start), offset);
				Point end = add(transform(line.end), offset);
				float d = get_ray_to_line_segment_intersection({ (float)x, (float)y }, { 1, 0 }, start, end);

				if (d != -1) inside = !inside;
			}

			if (inside) setPixel(image_data, x, y, width, color);
		}
	}
}

float sqr(float x) { return x * x; }
float dist2(Point v, Point w) { return sqr(v.x - w.x) + sqr(v.y - w.y); }
float dist_to_segment_squared(Point p, Point p1, Point p2) {
	float l2 = dist2(p1, p2);
	if (l2 == 0) return dist2(p, p1);
	float t = ((p.x - p1.x) * (p2.x - p1.x) + (p.y - p1.y) * (p2.y - p1.y)) / l2;
	t = fmax(0, fmin(1, t));
	return dist2(p, { p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y) });
}

void draw_glyph_stroke(unsigned char* image_data, LineSegment* lines, int line_count, float weight, Color color, Point offset, Point(*transform)(Point)) {
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			float min_dist = numeric_limits<float>::max();

			for (int l = 0; l < line_count; l++) {
				LineSegment line = lines[l];
				Point start = add(transform(line.start), offset);
				Point end = add(transform(line.end), offset);
				min_dist = min(min_dist, dist_to_segment_squared({ (float)x, (float)y }, start, end));
			}
			if (min_dist < weight)  setPixel(image_data, x, y, width, color);
		}
	}
}

Point scale_down(Point p) {
	float s = 8;
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

int main() {
	int image_data_size = sizeof(unsigned char) * channels * width * height;
	unsigned char* image_data = (unsigned char*)malloc(image_data_size);
	if (image_data == nullptr) {
		//cout << "BAD" << endl;
		return 0;
	}
	memset(image_data, 0, image_data_size);

	// unsigned char *data = stbi_load("test.png", &width, &height, &channels, STBI_rgb_alpha);


	//for (int x = 0; x < width / 2; x++) {
	//    for (int y = 0; y < height / 2; y++) {
	//        data[(y * width + x) * 4 + 0] = x;
	//        data[(y * width + x) * 4 + 1] = y;
	//    }
	//}

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			image_data[(r * width + c) * 4 + 3] = 255;
		}
	}


	int line_count = sizeof(lines_for_Question) / sizeof(LineSegment);

	 // draw_glyph(image_data, lines_for_A, line_count, { 255, 0, 0 }, { 20, 0 }, scale_down_and_italic, width);
	 // draw_glyph_stroke(image_data, lines_for_A, line_count, 30, { 0, 0, 255 }, { 20, 0 }, scale_down_and_italic, width);


	unsigned char* image_data_gpu;
	TRY_CUDA(cudaMalloc(&image_data_gpu, image_data_size));
	TRY_CUDA(cudaMemcpy(image_data_gpu, image_data, image_data_size, cudaMemcpyHostToDevice));

	LineSegment *lines_transformed = new LineSegment[line_count];
	for (int i = 0; i < line_count; i++) {
		lines_transformed[i].start = scale_down_and_italic(lines_for_Question[i].start);
		lines_transformed[i].end = scale_down_and_italic(lines_for_Question[i].end);
	}

		LineSegment* lines_gpu;
	TRY_CUDA(cudaMalloc(&lines_gpu, sizeof(lines_for_Question)));
	TRY_CUDA(cudaMemcpy(lines_gpu, lines_transformed, sizeof(lines_for_Question), cudaMemcpyHostToDevice));

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size(divide_ceil(width, BLOCK_SIZE), divide_ceil(height, BLOCK_SIZE));

	draw_glyph_gpu << <grid_size, block_size >> > (image_data_gpu, lines_gpu, line_count, { 0, 0, 255 }, { 600, 0 }, width);
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	
	//for (int i = 0; i < 5; i++) {
	//	draw_glyph_gpu << <grid_size, block_size >> > (image_data_gpu, lines_gpu, line_count, { 255, 0, 0 }, { (float) i * 600, 0 }, width);
	//	TRY_CUDA(cudaGetLastError());
	//	TRY_CUDA(cudaDeviceSynchronize());
	//}

	TRY_CUDA(cudaMemcpy(image_data, image_data_gpu, image_data_size, cudaMemcpyDeviceToHost));

	stbi_write_png("test_out.png", width, height, channels, image_data, width * channels);

	cout << "test" << endl;

	free(image_data);

	TRY_CUDA(cudaFree(image_data_gpu));
	TRY_CUDA(cudaFree(lines_gpu));

	return 0;
}
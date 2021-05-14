#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>
#include <algorithm>

#include <iostream>

#define TIME 101

namespace reference
{
	template <typename T>
	void acos(const T* arg, T* out, size_t count)
	{
		for (size_t i = 0; i < count; i++)
		{
			out[i] = std::acos(arg[i]);
		}
	}
}

float get_rand_range(const float min, const float max) {
    return min + (float)(rand()) / ((float)(RAND_MAX/(max-min)));
}

bool acos (const float *arg, float *out, unsigned int size)
{
	unsigned int i, j;
	float32x4_t copy, x, res;
	float32x4_t v1, v2;
	uint32x4_t mask1, mask2, mask;
	for(i = 0; i < size / 4; i++)
	{
		//Loading 4 elements
		copy = vld1q_f32(arg);
		
		//Checking compliance with the interval [-1; 1]
		v1 = vmovq_n_f32(-1.0f);
		v2 = vmovq_n_f32(1.0f);
		mask1 = vcltq_f32(copy, v1);
		mask2 = vcgtq_f32(copy, v2);
		mask = vorrq_u32(mask1, mask2);
		for (j = 0; j < 4; j++)
			if(mask[j] != 0)
				return false;
		
		//Replacing -x with x
		mask = vcltzq_f32(copy);
		x = vmulq_n_f32(copy, -1.0f);
		x = vbslq_f32(mask, x, copy);
		
		//arccos(0) = 1.570796326
		mask = vceqzq_f32(copy);
		v1 = vmovq_n_f32(1.570796326f);
		res = vbslq_f32(mask, v1, res);
		
		//Calculating the arctg using piecewise linear approximation
		
		v2 = vmovq_n_f32(0.0625f); //if (0 < x < 0.0625)
		mask1 = vcgtzq_f32(x);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.5707963268f);
		v2 = vmovq_n_f32(-1.0006521888f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.0625f); //if (0.0625 <= x < 0.125)
		v2 = vmovq_n_f32(0.125f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.5710426344f);
		v2 = vmovq_n_f32(-1.0045931104f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.125f); //if (0.125 <= x < 0.1875)
		v2 = vmovq_n_f32(0.1875f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.57216949868f);
		v2 = vmovq_n_f32(-1.01360802464f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.1875f); //if (0.1875 <= x < 0.25)
		v2 = vmovq_n_f32(0.25f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.57412376114f);
		v2 = vmovq_n_f32(-1.02403075776f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.25f); //if (0.25 <= x < 0.3125)
		v2 = vmovq_n_f32(0.3125f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.5786898669f);
		v2 = vmovq_n_f32(-1.0422951808f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.3125f); //if (0.3125 <= x < 0.375)
		v2 = vmovq_n_f32(0.375f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.5858379759f);
		v2 = vmovq_n_f32(-1.0651691296f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.375f); //if (0.375 <= x < 0.4375)
		v2 = vmovq_n_f32(0.4375f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.5969184741f);
		v2 = vmovq_n_f32(-1.0947171248f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.4375f); //if (0.4375 <= x < 0.5)
		v2 = vmovq_n_f32(0.5f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.6134549976f);
		v2 = vmovq_n_f32(-1.1325148928f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.5f); //if (0.5 <= x < 0.5625)
		v2 = vmovq_n_f32(0.5625f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.63765868f);
		v2 = vmovq_n_f32(-1.1809222576f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.5625f); //if (0.5625 <= x < 0.625)
		v2 = vmovq_n_f32(0.625f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.6729159559f);
		v2 = vmovq_n_f32(-1.2436018592f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.625f); //if (0.625 <= x < 0.6875)
		v2 = vmovq_n_f32(0.6875f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.7247571189f);
		v2 = vmovq_n_f32(-1.32654772f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.6875f); //if (0.6875 <= x < 0.75)
		v2 = vmovq_n_f32(0.75f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.802990011f);
		v2 = vmovq_n_f32(-1.4403410176f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.75f); //if (0.75 <= x < 0.8125)
		v2 = vmovq_n_f32(0.8125f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(1.9271233582f);
		v2 = vmovq_n_f32(-1.6058521472f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.8125f); //if (0.8125 <= x < 0.875)
		v2 = vmovq_n_f32(0.875f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(2.1434722065f);
		v2 = vmovq_n_f32(-1.8721276528f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask, v1, res);
		
		v1 = vmovq_n_f32(0.875f); //if (0.875 <= x < 0.9375)
		v2 = vmovq_n_f32(0.9375f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcltq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(2.6045108307f);
		v2 = vmovq_n_f32(-2.3990289376f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask1, v1, res);
		
		v1 = vmovq_n_f32(0.9375f); //if (0.9375 <= x <= 1)
		v2 = vmovq_n_f32(1.0f);
	    mask1 = vcgeq_f32(x, v1);
		mask2 = vcleq_f32(x, v2);
		mask = vandq_u32(mask1, mask2);
		v1 = vmovq_n_f32(5.6867392272f);
		v2 = vmovq_n_f32(-5.6867392272f);
		v1 = vmlaq_f32(v1, v2, x);
		res = vbslq_f32(mask1, v1, res);
		
		
		// arccos(-x) = PI - arccos(x)
		mask = vcltzq_f32(copy);
		v1 = vmovq_n_f32(3.141592653f);
		v1 = vsubq_f32(v1, res);
		res = vbslq_f32(mask, v1, res);
		
		vst1q_f32(out, res);
		arg += 4;
		out += 4;
	}
	return true;
}


int main(int argc, char **argv)
{
	//srand(time(NULL));
	unsigned int size = 100000;
	float * arr = (float *)(malloc(sizeof(float) * size));
	float * arr_res_seq = (float*)(malloc(sizeof(float) * size));
	float * arr_res_neon = (float*)(malloc(sizeof(float) * size));
	float * time = (float*) malloc(TIME * sizeof(float));
	double start;
	double end;
	
	//for(unsigned int i = 0; i < size; i++) {
	//	arr[i] = get_rand_range(-1.0f, -0.7f);
	//	printf("%f \n\n", arr[i]);
	//}
	
	for(unsigned int i = 0; i < size; i++) {
		arr[i] = -1.0f + 2.0f / (float)size * i;
		//printf("%f \n\n", arr[i]);
	}
	
	for (size_t j = 0; j < TIME; j++){
		start = omp_get_wtime( );
		reference::acos(arr, arr_res_seq, size);
		end = omp_get_wtime( );
		//for(unsigned int i = 0; i < size; i++) {
		//	printf("%f \n", arr_res_seq[i]);
		//}
		if (j != 0 ) time[j-1] = end - start;
		
	}
	std::sort(time, time + TIME);
	printf("Time seq  = %.16g\n", time[TIME/2]);
	
	
	for (size_t j = 0; j < TIME; j++){
		start = omp_get_wtime( );
		acos(arr, arr_res_neon, size);
		end = omp_get_wtime( );
		if (j != 0 ) time[j-1] = end - start;
		
	}
	std::sort(time, time + TIME);
	printf("Time neon = %.16g\n", time[TIME/2]);
	
	float max_error = 0.0f;
	for(unsigned int i = 0; i < size; i++) {
		max_error = std::abs(arr_res_seq[i] - arr_res_neon[i]) > max_error ? std::abs(arr_res_seq[i] - arr_res_neon[i]) : max_error;
		//printf("%f \n", arr_res_neon[i]);
	}
	printf("Max error = %.16g\n", max_error);
	
    return 0;
}
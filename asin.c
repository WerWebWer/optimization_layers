// g++ alex.c -fopenmp -lm && ./a.out

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <arm_neon.h>
#include <algorithm>

#define NUM_TREADS 4
#define N 100 * 1000 * 1000
#define TIME 101
#define uint unsigned int

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // M_PI

namespace reference {
	template <typename T>
	void asin(const T* arg, T* out, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			out[i] = std::asin(arg[i]);
		}
	}
}// namespace reference

void add_seq(uint *a, uint *b, uint *c, uint size){
	int i = 0;
	for (i = 0; i < size; ++i)
		c[i] = a[i] + b[i];
}

void add_par(uint *a, uint *b, uint *c, uint size){
	unsigned int i;
	uint32x4_t Q0, Q1, Q2;
	for(i=0; i<(size/4); i++) {
		Q0=vld1q_u32(a);
		Q1=vld1q_u32(b);
		Q2=vaddq_u32(Q0,Q1);
		vst1q_u32(c,Q2);
		a+=4;
		b+=4;
		c+=4;
	}
}

/** Sin polynomial coefficients */
constexpr float te_sin_coeff2 = 0.166666666666f; // 1/(2*3)
constexpr float te_sin_coeff3 = 0.05f;           // 1/(4*5)
constexpr float te_sin_coeff4 = 0.023809523810f; // 1/(6*7)
constexpr float te_sin_coeff5 = 0.013888888889f; // 1/(8*9)

/** ArcSin polynomial coefficients */
constexpr float te_asin_coeff2 = 0.16666666666f; // 1/6
constexpr float te_asin_coeff3 = 0.075f;         // 3/40
constexpr float te_asin_coeff4 = 0.04464285714f; // 5/112
constexpr float te_asin_coeff5 = 0.03038194444f; // 35/1152
constexpr float te_asin_coeff6 = 0.02237215909f; // 63/2816

inline float32x4_t vsin_f32(float32x4_t val){
    const float32x4_t pi_v   = vdupq_n_f32(M_PI);
    const float32x4_t pio2_v = vdupq_n_f32(M_PI / 2);
    const float32x4_t ipi_v  = vdupq_n_f32(1 / M_PI);

    //Find positive or negative
    const int32x4_t  c_v    = vabsq_s32(vcvtq_s32_f32(vmulq_f32(val, ipi_v)));
    const uint32x4_t sign_v = vcleq_f32(val, vdupq_n_f32(0));
    const uint32x4_t odd_v  = vandq_u32(vreinterpretq_u32_s32(c_v), vdupq_n_u32(1));

    uint32x4_t neg_v = veorq_u32(odd_v, sign_v);

    //Modulus a - (n * int(a*(1/n)))
    float32x4_t      ma    = vsubq_f32(vabsq_f32(val), vmulq_f32(pi_v, vcvtq_f32_s32(c_v)));
    const uint32x4_t reb_v = vcgeq_f32(ma, pio2_v);

    //Rebase a between 0 and pi/2
    ma = vbslq_f32(reb_v, vsubq_f32(pi_v, ma), ma);

    //Taylor series
    const float32x4_t ma2 = vmulq_f32(ma, ma);

    //2nd elem: x^3 / 3!
    float32x4_t elem = vmulq_f32(vmulq_f32(ma, ma2), vdupq_n_f32(te_sin_coeff2));
    float32x4_t res  = vsubq_f32(ma, elem);

    //3rd elem: x^5 / 5!
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff3));
    res  = vaddq_f32(res, elem);

    //4th elem: x^7 / 7!float32x2_t vsin_f32(float32x2_t val)
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff4));
    res  = vsubq_f32(res, elem);

    //5th elem: x^9 / 9!
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff5));
    res  = vaddq_f32(res, elem);

    //Change of sign
    neg_v = vshlq_n_u32(neg_v, 31);
    res   = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(res), neg_v));
    return res;
}

inline float32x4_t vasin_f32(float32x4_t val){
    //Modulus a - (n * int(a*(1/n)))
    float32x4_t ma = val;

    //Taylor series
    const float32x4_t ma2 = vmulq_f32(ma, ma);

    //2nd elem: x^3 / 6
    float32x4_t elem = vmulq_f32(vmulq_f32(ma, ma2), vdupq_n_f32(te_asin_coeff2));
    float32x4_t res  = vaddq_f32(ma, elem);

    //3rd elem: 3 x^5 / 40
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_asin_coeff3));
    res  = vaddq_f32(res, elem);

    //4th elem: 5 x^7 / 112
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_asin_coeff4));
    res  = vaddq_f32(res, elem);

    //5th elem: 35 x^9 / 1152
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_asin_coeff5));
    res  = vaddq_f32(res, elem);

    //6th elem: 63 x^11 / 2816
    /*elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_asin_coeff6));
    res  = vaddq_f32(res, elem);*/
    return res;
}

void vsin(float *data, float * res, size_t size){
	
	float32x4_t Q0, Q1;
	
	for (size_t i = 0; i < size/4; i++){
		Q0 = vld1q_f32(data);
		Q1 = vsin_f32(Q0);
		vst1q_f32(res, Q1);
		data += 4;
		res += 4;
	}
	
}

void vasin(float *data, float * res, size_t size){
	
	float32x4_t Q0, Q1;
	
	for (size_t i = 0; i < size/4; i++){
		Q0 = vld1q_f32(data);
		Q1 = vasin_f32(Q0);
		vst1q_f32(res, Q1);
		data += 4;
		res += 4;
	}
	
}

int main(int argc, char** argv){
	float *data1 = NULL;
	float *data2 = NULL;
	float *res_seq = NULL;
	float *res_par = NULL;
	float *time = NULL;
	float error = 0.0;
	float error_loc = 0.0;
	size_t cnt = 0;
	double t1, t2, t_seq, t_par;
	size_t i = 0;
	
	omp_set_num_threads(NUM_TREADS);
	
	printf("Hello!\n\r");
	
	printf("%d\n", N);
	printf("%d\n\n", TIME);
	
	data1 = (float*) malloc(N * sizeof(float));
	//data2 = (float*) malloc(N * sizeof(float));
	res_seq = (float*) malloc(N * sizeof(float));
	res_par = (float*) malloc(N * sizeof(float));
	time = (float*) malloc(TIME * sizeof(float));
	
	for (i = 0; i < N; ++i){
		data1[i] = 0 +  0.5 / (float)N * (float)i;
		//data2[i] = - (float)i / (float)N;
		res_seq[i] = 0;
		res_par[i] = 0;
		//printf("data[%d] = %d | %d\n", i, data1[i], data2[i]);
		//printf("data[%d] = %d/%d = %.10lf\n", i, i, N, data1[i]);
	}
	
	printf("\n");
	
// sequestion
	for (size_t j = 0; j < TIME; j++){
		t1 = omp_get_wtime();
		
		for (i = 0; i < N; i++){
			res_seq[i] = std::asin(data1[i]);
			
		};
		
		t2 = omp_get_wtime();
		t_seq = t2-t1;
		if (j != 0 ) time[j-1] = t_seq;
		
	}
	std::sort(time, time + TIME);
	printf("time seq = %f\n", time[TIME / 2]);
	printf("time seq = %.16g\n", time[TIME / 2]);
	
	printf("\n------------\n\n");
	
// parallel
	for (size_t j = 0; j < TIME; j++){
		t1 = omp_get_wtime();

		vasin(data1, res_par, N);
		
		t2 = omp_get_wtime();
		t_par = t2-t1;
		if (j != 0 ) time[j-1] = t_seq;
	}
	std::sort(time, time + TIME);
	printf("time vec = %f\n", time[TIME / 2]);
	printf("time vec = %.16g\n", time[TIME / 2]);
	
	printf("\n------------\n\n");
	
// result
	/*printf("\n   i   |       seq     |       vec     |    error    \n");
	for (i = 0; i < N; ++i){
		if (i <= 9) printf(" ");
		if (i <= 99) printf(" ");
		printf("%d/%d | ", i, N);
		if (res_seq[i] >= 0)
			printf(" %.10lf", res_seq[i]);
		else 
			printf("%.10lf", res_seq[i]);
		printf(" | ");
		if (res_par[i] >= 0)
			printf(" %.10lf", res_par[i]);
		else 
			printf("%.10lf", res_par[i]);
		error_loc = std::abs(res_par[i] - res_seq[i]);
		printf(" | %.10lf", error_loc);
		if (error < error_loc){
			error = error_loc;
			cnt = i;
		}
		printf("\n");
	}*/
	
// error
	for (i = 0; i < N; ++i){
		error_loc = std::abs(res_par[i] - res_seq[i]);
		if (error < error_loc){
			error = error_loc;
			cnt = i;
		}
	}
	
	printf("Max error = [%lu] %.10lf\n", cnt, error);
	
	printf("\ntime seq = %.10lf\n", t_seq);
	printf("time seq = %.16g\n", t_seq);
	printf("\n");
	printf("time vec = %.10lf\n", t_par);
	printf("time vec = %.16g\n", t_par);
	
	printf("\n\r");
	
	free(data1);
	//free(data2);
	free(res_seq);
	free(res_par);
	
	return 0;
}

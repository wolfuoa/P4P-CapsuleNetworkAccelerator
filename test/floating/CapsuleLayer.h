#ifndef _CAPSULELAYER_H_
#define _CAPSULELAYER_H_

#include <stdint.h>
#define NUM_CAPSULE 10
#define DIM_CAPSULE 16
#define ROUTINGS 3
#define INPUT_NUM_CAPSULE 1152
#define INPUT_DIM_CAPSULE 8
#define BATCH_SIZE 1
#define INPUT_HAT_DIM_0 1
#define BC_DIM_0_1 1

void getMultiply(float *matA, float *matB, float *matC);
void getInputHat(float *input, float *weights, float *output);
void matmulInputHat(float *matA, float *matB, float *matC, uint32_t startIndexA, uint32_t startIndexB, uint32_t startIndexC);
void matmulAgreement(float *matA, float *matB, float *matC, uint32_t startIndexA, uint32_t startIndexB, uint32_t startIndexC);
void getSoftmax(float *matA, float *matB);
void getSum(float *matA, float *matB);
void initialiseInputAndWeightArray();
void getSquash(float *matA, float *matB);
float getSquaredNorm(float *matA);
void getScale(float *matB);
void getSquaredNorm(float *matA, float *matB);
void getAgreement(float *inputHat, float *outputsSquash, float *agreement);
void getAdd(float *A, float *B);
float getMaxColumn(float *mat, uint32_t starting_index);
void run(float *input, float *weights, float *output);
void printSize();
void runInFile();

#endif

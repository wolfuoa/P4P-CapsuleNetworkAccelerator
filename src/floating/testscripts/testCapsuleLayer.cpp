#include <iostream>
#include <math.h>
#include <stdint.h>
#include <tuple>
#include <fstream>
#include "CapsuleLayer.h"
#include <limits>
using namespace std;

uint32_t weightSize = NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE * INPUT_DIM_CAPSULE;
uint32_t inputSize = BATCH_SIZE * INPUT_NUM_CAPSULE * INPUT_DIM_CAPSULE;
uint32_t outputsSize = BATCH_SIZE * NUM_CAPSULE * DIM_CAPSULE;

// weight tensor has shape [NUM_CAPSULE, INPUT_NUM_CAPSULE, DIM_CAPSULE, INPUT_DIM_CAPSULE]
float weightsTest[NUM_CAPSULE * INPUT_NUM_CAPSULE * DIM_CAPSULE * INPUT_DIM_CAPSULE];
// input tensor has shape [BATCH_SIZE, INPUT_NUM_CAPSULE, INPUT_DIM_CAPSULE]
float inputTest[BATCH_SIZE * INPUT_NUM_CAPSULE * INPUT_DIM_CAPSULE];
typedef std::numeric_limits< float > dbl;

int main() {
    for (uint32_t i = 0; i < weightSize; i++) { weightsTest[i] = i % 100; }
    for (uint32_t i = 0; i < inputSize; i++) { inputTest[i] = i % 100; }

    cout.precision(dbl::max_digits10);
    float *outputsSumTemp;

    run(inputTest, weightsTest, outputsSumTemp);

    cout << "Finished!" << endl;
    ofstream output;
    output.open ("capsuleLayerOutputs/outputs_squash_c.txt");
    output.precision(dbl::max_digits10);

    for (int i = 0; i < outputsSize; i++) {
        output << *(outputsSumTemp + i) << endl;
    }

    output.close();

    // compare the two files
    // fstream goldenOutput;
    // fstream layerOutput;

    // goldenOutput.open ("golden.txt", ios::in);
    // layerOutput.open("layerOutput.txt", ios::in);

    // string goldenVal;
    // string outputVal;

    // while(getline(goldenOutput, goldenVal)) {
    //     getline(layerOutput, outputVal);
    //     if (outputVal != goldenVal) {
    //         cout << "NOT MATCH" << endl;
    //         return 1;
    //     }
    // }

    // layerOutput.close();
    // goldenOutput.close();


    return 0;
}
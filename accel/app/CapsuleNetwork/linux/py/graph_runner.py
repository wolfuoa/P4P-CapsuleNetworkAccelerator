'''
Copyright 2022 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations un:der the License.
'''

import cv2
import numpy as np
import os
import xir
import argparse
import vitis_ai_library
from mnist import MNIST

supported_ext = [".jpg", ".jpeg", ".png"]


def get_imagefiles():
    # the data, shuffled and split between train and test sets
    mndata = MNIST('./MNIST')
    (x_test, y_test) = mndata.load_testing()

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    return (x_test, y_test)


def preprocess_tf2_custom_op(imgfiles, begidx, input_tensor_buffers):
    input_tensor = input_tensor_buffers[0].get_tensor()
    batchsize = input_tensor.dims[0]
    height = input_tensor.dims[1]
    width = input_tensor.dims[2]

    fix_pos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")
    scale = 2**fix_pos

    for i in range(batchsize):
        img = imgfiles[begidx + i]
        img = (img * scale).astype(np.int8)

        input_data = np.asarray(input_tensor_buffers[0])
        input_data[i][:] = img

def printAccuracy(y_test, pred):
    correct = 0
    
    for i in range(pred.size):
        if (pred[i] == y_test[i]):
            correct = correct + 1

    print(correct/pred.size)
    
def app(model, num_images):
    (x_test, y_test) = get_imagefiles()
    assert(num_images <= len(y_test))

    g = xir.Graph.deserialize(model)
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)

    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    #batchsize = input_tensor_buffers[0].get_tensor().dims[0]

    pred = np.zeros(num_images)
    for i in range(num_images):
        #begin_idx = i * batchsize
        preprocess_tf2_custom_op(x_test, i, input_tensor_buffers)

        v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
        runner.wait(v)

        output_data = np.array(output_tensor_buffers[0])
        pred[i] = np.argmax(output_data[0])
    
    printAccuracy(y_test, pred)

    del runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str, help='xmodel name')
    ap.add_argument('num_images', type=int, help='number of MNIST testing images to run prediction on')

    args = ap.parse_args()

    print('Command line options:')
    print(' model      : ', args.model)
    print(' num_images : ', args.num_images)

    app(args.model, args.num_images)


if __name__ == '__main__':
    main()
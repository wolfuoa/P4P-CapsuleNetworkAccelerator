from torch import quantize_per_channel
import os
import tensorflow as tf 
from tensorflow_model_optimization.quantization.keras import vitis_inspect, vitis_quantize
from keras import Model
from capsulenet import *
from capsulelayers import Length, Mask, CapsuleLayer, SquashLayer

def quantize_model(file_path, save_path):

    # load MNIST for quantization calibration
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # load model to quantize
    new_model = tf.keras.models.load_model(file_path, custom_objects={'Length': Length, 'CapsuleLayer': CapsuleLayer, 'SquashLayer': SquashLayer})

    # quantize model
    quantizer = vitis_quantize.VitisQuantizer(new_model, quantize_strategy='pof2s', custom_objects={'Length': Length, 'CapsuleLayer': CapsuleLayer, 'SquashLayer': SquashLayer})
    quantized_model = quantizer.quantize_model(calib_dataset = x_test, add_shape_info = True, separate_conv_act = False)
    quantized_model.save(os.path.join(save_path, 'quantized_partial_model.h5'))

    # dump quantize results
    vitis_quantize.VitisQuantizer.dump_model( model=quantized_model, dataset=x_test[0:1], output_dir="./dump_results", dump_float=True)

    # (experimental) inspector code
    # inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B4096");
    # inspector.inspect_model(model, plot=True, plot_file = "model.svg", dump_results=True, dump_results_file="inspect_results.txt", verbose=1)

if __name__ == "__main__":
    import argparse

    # get user input on w
    parser = argparse.ArgumentParser(description="Run Vitis AI quantization")
    parser.add_argument('--file_path', default=os.path.join('models', 'partial_model.h5'),
                        help="Path of model to quantize. Default is models/partial_model.h5")
    parser.add_argument('--save_path', default='models/quant',
                        help="Path to save model. Default is models/quant")
    args = parser.parse_args()
    print(args)

    quantize_model(args.file_path, args.save_path)

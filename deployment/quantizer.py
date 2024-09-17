from torch import quantize_per_channel
import os
import tensorflow as tf 
from tensorflow_model_optimization.quantization.keras import vitis_inspect, vitis_quantize
from keras import Model
from capsulenet import *
from capsulelayers import Length, Mask, CapsuleLayer, SquashLayer
from createNoReconstructionModel import remove_reconstruction

def quantize_model(file_path, save_path):

    # load MNIST for quantization calibration
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # load model to quantize
    full_model = tf.keras.models.load_model(file_path, custom_objects={'Length': Length, 'CapsuleLayer': CapsuleLayer, 'SquashLayer': SquashLayer})

    # Extract the all layers excluding digitCaps
    partial_model = Model(inputs=full_model.input, outputs=full_model.layers[4].output)

    # Quantizing
    quantizer = vitis_quantize.VitisQuantizer(partial_model, quantize_strategy='pof2s', custom_objects={'Length': Length, 'CapsuleLayer': CapsuleLayer, 'SquashLayer': SquashLayer})
    quantized_model = quantizer.quantize_model(calib_dataset=x_test, add_shape_info=True, separate_conv_act=False)

    # Save the quantized partial model
    quantized_model.save(os.path.join(save_path, 'no_digitcaps_model.h5'))

    # Dump quantization results
    vitis_quantize.VitisQuantizer.dump_model(model=quantized_model, dataset=x_test[0:1], output_dir="./dump_results", dump_float=True)

    # (experimental) inspector code
    # inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B4096");
    # inspector.inspect_model(model, plot=True, plot_file = "model.svg", dump_results=True, dump_results_file="inspect_results.txt", verbose=1)

if __name__ == "__main__":
    import argparse

    # get user input on w
    parser = argparse.ArgumentParser(description="Run Vitis AI quantization")
    parser.add_argument('--file_path', default=os.path.join('TF_models', 'no_reconstruction_eval_trained_model.h5'),
                        help="Path of model to quantize. Default is TF_models/no_reconstruction_eval_model.h5")
    parser.add_argument('--save_path', default='quantized_va',
                        help="Path to save model. Default is quantized_va")
    args = parser.parse_args()
    print(args)

    quantize_model(args.file_path, args.save_path)
import numpy as np
import os
from keras import layers, models, optimizers, Model
from capsulenet import *
from capsulelayers import Length, Mask, CapsuleLayer, SquashLayer
# the data, shuffled and split between train and test sets

def remove_reconstruction(evaluate = False):

    # load MNIST data set
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # load pre-trained MNIST capsnet
    model, eval_model, manipulate_model = CapsNet(input_shape=x_test.shape[1:],
                                            n_class=len(np.unique(np.argmax(y_train, 1))),
                                            routings=3)
    eval_model.load_weights(os.path.join('models', 'trained_model.h5'))
    eval_model.summary()

    layer_name = 'capsnet'

    # remove reconstruction and save model
    model_no_reconstruction= Model(inputs=eval_model.input, outputs=eval_model.get_layer(layer_name).output)
    model_no_reconstruction.summary()

    # if specified, test model on MNIST testing dataset
    if evaluate == True: 
        model_no_reconstruction.compile(optimizer=optimizers.Adam(0.005),
                      loss=[margin_loss, 'mse'],
                      metrics={'capsnet': 'accuracy'})    
        model_no_reconstruction.evaluate(x_test, y_test)

    layer_name = 'primarycap_squash'

    # remove reconstruction and save model
    partial_model= Model(inputs=eval_model.input, outputs=eval_model.get_layer(layer_name).output)
    partial_model.summary()

    return partial_model


if __name__ == "__main__":
    import argparse

    # get user input to test no reconstruction model on MNIST testing set
    parser = argparse.ArgumentParser(description="Create Capsule Network without DigitCaps.")
    parser.add_argument('-t', '--test', action='store_true',
                        help="Test the model on testing dataset")
    parser.add_argument('--save_dir', default="models",
                        help="Directory to save generated model. Default is in /models.")
    args = parser.parse_args()
    print(args)

    model = remove_reconstruction(args.test)

    # generate saved model
    if not args.save_dir:
        model.save('partial_model.h5')
    else:
        model.save(os.path.join(args.save_dir, 'partial_model.h5'))

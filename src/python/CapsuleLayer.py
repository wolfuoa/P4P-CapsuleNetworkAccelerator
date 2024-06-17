from cv2 import transpose
import numpy as np
from scipy import special
import os

class CapsuleLayer:

    def __init__(self):
        pass
    
    def squash(self, vectors, axis=-1):
        s_squared_norm = np.sum(np.square(vectors), axis, keepdims=True)
        # update epilson
        scale = s_squared_norm / ((1 + s_squared_norm) * np.sqrt(s_squared_norm + 1e-7))
        return scale * vectors

    def calculate(self, output, input):
        if len(input) == 0:
            return

        self.num_capsule = 10
        self.dim_capsule = 16
        self.routings = 3
        self.input_num_capsule = 1152
        self.input_dim_capsule = 8

        # convert input to np arrays
        np_input = np.array(input[0], copy= False)
        np_output = np.array(output, copy = False)
        self.W = np.array(input[1], copy=False)

        inputs_expand = np.expand_dims(np_input, 1)
        inputs_tiled  = np.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled  = np.expand_dims(inputs_tiled, 4)

        # could be wrong - had to change map_fn to map 
        inputs_hat = np.array(list(map(lambda x: np.matmul(self.W, x), inputs_tiled)))  
        b = np.zeros(shape=[np.shape(inputs_hat)[0], self.num_capsule, 
                            self.input_num_capsule, 1, 1])
        # with open(os.path.join('capsule_layer_outputs', 'weights_p.txt'), 'w') as my_file: 
        #     for w_1 in self.W:
        #         for w_2 in w_1: 
        #             for w_3 in w_2: 
        #                 np.savetxt(my_file, np.round(w_3))

        # with open(os.path.join('capsule_layer_outputs', 'input_p.txt'), 'w') as my_file: 
        #     for i_1 in np_input:
        #         for i_2 in i_1: 
        #             np.savetxt(my_file, np.round(i_2))

        # with open(os.path.join('capsule_layer_outputs', 'inputs_hat_p.txt'), 'w') as my_file: 
        #         for a in inputs_hat:
        #             for aa in a:
        #                 for aaa in aa:
        #                         np.savetxt(my_file, aaa)

        transpose_inputs = inputs_hat.transpose((0, 1, 2, 4, 3))

        assert self.routings > 0, 'The routings should be > 0.'

        for i in range(self.routings):
            # Apply softmax to the axis with `num_capsule`
            c = special.softmax(b, axis=1)
            # with open(os.path.join('capsule_layer_outputs', 'outputs_softmax_p.txt'), 'w') as my_file: 
            #     for n in c:
            #         for j in n:
            #             for k in j:
            #                 for m in k:
            #                     np.savetxt(my_file, np.around(m, 1))

            # Compute the weighted sum of all the predicted output vectors.
            #  c.shape =  [batch_size, num_capsule, input_num_capsule, 1, 1]
            #  inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule,1]
            # The function `multiply` will broadcast axis=3 in c to dim_capsule.
            #  outputs.shape=[None, num_capsule, input_num_capsule, dim_capsule, 1]
            # Then sum along the input_num_capsule
            #  outputs.shape=[None, num_capsule, 1, dim_capsule, 1]
            # Then apply squash along the dim_capsule
            outputs = np.multiply(c, inputs_hat)
            # print(outputs.size)
            # with open(os.path.join('capsule_layer_outputs', 'outputs_multiply_p.txt'), 'w') as my_file: 
            #     for n in outputs:
            #         for j in n:
            #             for k in j:
            #                 for m in k:
            #                     np.savetxt(my_file, np.around(m, 1))
            np.set_printoptions(threshold=np.inf)

            outputs = np.sum(outputs, axis=2, keepdims=True)

            # with open(os.path.join('capsule_layer_outputs', 'outputs_sum_p.txt'), 'w') as my_file: 
            #     for a in outputs:
            #         for aa in a:
            #             for aaa in aa:
            #                     np.savetxt(my_file, aaa)

            outputs = self.squash(outputs, axis=-2)  # [None, 10, 1, 16, 1]
            
            
            if i < self.routings - 1:
            # Update the prior b.s
            #  outputs.shape =  [None, num_capsule, 1, dim_capsule, 1]
            #  inputs_hat.shape=[None,num_capsule,input_num_capsule,dim_capsule,1]
            # Multiply the outputs with the weighted_inputs (inputs_hat) and add  
            # it to the prior b.  
                outputs_tiled = np.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = np.matmul(transpose_inputs, outputs_tiled)
                b = np.add(b, agreement)

                # with open(os.path.join('capsule_layer_outputs', 'outputs_b_p.txt'), 'w') as my_file: 
                #     for a in b:
                #         for aa in a:
                #             for aaa in aa:
                #                     np.savetxt(my_file, aaa)
                # with open(os.path.join('capsule_layer_outputs', 'outputs_agreement_p.txt'), 'w') as my_file: 
                #     for a in b:
                #         for aa in a:
                #             for aaa in aa:
                #                     np.savetxt(my_file, aaa)
        # End: Routing algorithm ------------------------------------------------#
        # Squeeze the outputs to remove useless axis:
        #  From  --> outputs.shape=[None, num_capsule, 1, dim_capsule, 1]
        #  To    --> outputs.shape=[None, num_capsule,    dim_capsule]
        outputs = np.squeeze(outputs, (2, 4))

        with open(os.path.join('capsule_layer_outputs', 'outputs_squash_p.txt'), 'w') as my_file: 
            for a in outputs:
                for aa in a:
                    np.savetxt(my_file, aa)
                        
        np.copyto(np_output, outputs)                    
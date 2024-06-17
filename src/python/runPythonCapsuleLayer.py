from CapsuleLayer import *
import numpy as np

num_capsule = 10
dim_capsule = 16
routings = 3
input_num_capsule = 1152
input_dim_capsule = 8
batch_size = 1

weights = np.ndarray(shape = (num_capsule, input_num_capsule, dim_capsule, input_dim_capsule))
input_val = np.ndarray(shape = (batch_size, input_num_capsule, input_dim_capsule))

# print(weights.size)
sum = 0
for x in range(num_capsule):
    for y in range(input_num_capsule):
        for z in range(dim_capsule):
            for m in range(input_dim_capsule):
                weights[x][y][z][m] = sum % 100
                sum = sum + 1

sum = 0
for x in range(batch_size):
    for y in range(input_num_capsule):
        for m in range(input_dim_capsule):
            input_val[x][y][m] = sum % 100
            sum = sum + 1

caps = CapsuleLayer()

output = np.ndarray(shape = (batch_size, num_capsule, dim_capsule))

caps.calculate((output), (input_val, weights))
import numpy as np
import numpy as np

class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        # Assume input is square for simplicity
        self.max_indices = None  # This will now be an array of tuples

    def forward(self, input):
        self.input_shape = input.shape
        output_height = 1 + (input.shape[0] - self.pool_size) // self.stride
        output_width = 1 + (input.shape[1] - self.pool_size) // self.stride
        output = np.zeros((output_height, output_width))
        self.max_indices = np.zeros((output_height, output_width, 2), dtype=int)  # Store indices as (y, x)

        for y in range(0, input.shape[0] - self.pool_size + 1, self.stride):
            for x in range(0, input.shape[1] - self.pool_size + 1, self.stride):
                window = input[y:y+self.pool_size, x:x+self.pool_size]
                max_val = np.max(window)
                output[y // self.stride, x // self.stride] = max_val
                
                # Find the index of the maximum value in the window
                max_index = np.unravel_index(np.argmax(window, axis=None), window.shape)
                self.max_indices[y // self.stride, x // self.stride] = (y + max_index[0], x + max_index[1])

        return output

    def backward(self, d_loss):
        # Reshape d_loss to match the output shape of forward pass
        d_loss_reshaped = d_loss.reshape((1 + (self.input_shape[0] - self.pool_size) // self.stride,
                                          1 + (self.input_shape[1] - self.pool_size) // self.stride))
        d_input = np.zeros(self.input_shape)
        
        for y in range(d_loss_reshaped.shape[0]):
            for x in range(d_loss_reshaped.shape[1]):
                (i, j) = self.max_indices[y, x]
                d_input[i, j] += d_loss_reshaped[y, x]
                
        return d_input

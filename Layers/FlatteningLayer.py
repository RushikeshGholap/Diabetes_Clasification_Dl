import numpy as np
class FlatteningLayer:
    def __init__(self):
        self.original_shape = None  # Initialize original_shape

    def forward(self, input):
        # Check if the input is already a 2D array (batch_size, features)
        if input.ndim > 2:
            # Flatten the input while keeping the batch size dimension
            self.original_shape = input.shape  # Store original shape for backward pass
            batch_size = input.shape[0]
            flattened_size = np.prod(input.shape[1:])  # Product of dimensions except for batch size
            return input.reshape(batch_size, flattened_size)
        elif input.ndim == 1:
            # If it's a single input vector without batch dimension, reshape it to (1, num_features)
            return input.reshape(1, -1)
        else:
            # If input is already 2D, return it as is
            return input

    def backward(self, grad_output):
        # If necessary, reshape the gradient to match the original input shape
        # This is a placeholder; actual implementation depends on how you handle the backward pass
        return grad_output.reshape(self.original_shape)

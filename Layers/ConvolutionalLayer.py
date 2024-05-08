import numpy as np

class ConvolutionalLayer:
    def __init__(self, kernel_width, kernel_height, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.kernel = np.random.randn(kernel_height, kernel_width)
        self.initial_kernel = self.kernel.copy()
        # Adam optimizer parameters
        self.m_kernel = np.zeros_like(self.kernel)
        self.v_kernel = np.zeros_like(self.kernel)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0


    def forward(self, input):
        self.input = input
        # Assuming the input is reshaped to (1, 40, 40) and treating the first dimension as channels
        if input.ndim == 3:
            # No padding is needed to achieve 32x32 output with a 9x9 kernel and stride of 1
            return self.crossCorrelate2D(input[0], self.kernel)  # Process the single-channel
        else:
            raise ValueError("Unsupported input dimensions")

    def crossCorrelate2D(self, input, kernel):
        kernel_height, kernel_width = kernel.shape
        output_height = input.shape[0] - kernel_height + 1
        output_width = input.shape[1] - kernel_width + 1
        
        # Initialize output without assuming additional channel dimensions
        output = np.zeros((output_height, output_width))
        
        for y in range(output_height):
            for x in range(output_width):
                output[y, x] = np.sum(input[y:y+kernel_height, x:x+kernel_width] * kernel)
        
        return output


    def backward(self, input, grad_output):
        # input = self.conv_layer.input
        # Calculate gradient of loss w.r.t. kernel (placeholder, assuming you have this logic)
        self.grad_kernel = self.calculate_kernel_gradient(input, grad_output)
        return self.grad_kernel

    def calculate_kernel_gradient(self, input, grad_output):
        # print(grad_output)
        kernel_height, kernel_width = self.kernel.shape
        grad_kernel = np.zeros_like(self.kernel)
        for y in range(grad_kernel.shape[0]):
            for x in range(grad_kernel.shape[1]):
                mat_cal = np.sum(input[:, y:y+grad_output.shape[0], x:x+grad_output.shape[1]] * grad_output)
                grad_kernel[y, x] = mat_cal
        
        return grad_kernel

    def updateWeights(self,learning_rate):
        self.t += 1  # Increment timestep for bias correction

        # Update first moment estimate
        self.m_kernel = self.beta1 * self.m_kernel + (1 - self.beta1) * self.grad_kernel
        
        # Update second raw moment estimate
        self.v_kernel = self.beta2 * self.v_kernel + (1 - self.beta2) * (self.grad_kernel ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_kernel = self.m_kernel / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat_kernel = self.v_kernel / (1 - self.beta2 ** self.t)

        mat_cal = learning_rate * m_hat_kernel / (np.sqrt(v_hat_kernel) + self.epsilon)
        # print('kernel before ,with',mat_cal)
        # Update weights
        self.kernel -= mat_cal
        # print('kernel updated,with',self.kernel)



# class ConvolutionalLayer:
#     def __init__(self, kernel_width, kernel_height):
#         self.kernel = np.random.randn(kernel_height, kernel_width)
#         self.grad_kernel = np.zeros_like(self.kernel)  # Initialize gradient placeholder

#     def forward(self, input):
#         # Assuming the input is reshaped to (1, 40, 40) and treating the first dimension as channels
#         if input.ndim == 3:
#             # No padding is needed to achieve 32x32 output with a 9x9 kernel and stride of 1
#             return self.crossCorrelate2D(input[0], self.kernel)  # Process the single-channel
#         else:
#             raise ValueError("Unsupported input dimensions")

#     def crossCorrelate2D(self, input, kernel):
#         kernel_height, kernel_width = kernel.shape
#         output_height = input.shape[0] - kernel_height + 1
#         output_width = input.shape[1] - kernel_width + 1
        
#         # Initialize output without assuming additional channel dimensions
#         output = np.zeros((output_height, output_width))
        
#         for y in range(output_height):
#             for x in range(output_width):
#                 output[y, x] = np.sum(input[y:y+kernel_height, x:x+kernel_width] * kernel)
        
#         return output

#     def backward(self, d_out):
#         pass

#     def updateWeights(self,learning_rate):
#             self.learning_rate = learning_rate
#             # Update kernel based on calculated gradients
#             self.kernel -= learning_rate * self.grad_kernel

#     def calculate_kernel_gradient(input, grad_output):
#         # Placeholder for the actual gradient calculation logic
#         # It involves 'convolving' the input with the grad_output in a certain way
#         return np.random.randn(*self.kernel.shape) * 0.01  # Placeholder logic

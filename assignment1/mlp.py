import numpy as np


class TwoLayerMLP:
    def __init__(self, d_in, h, d_out):
        # Weight initialization
        self.W1 = np.random.randn(d_in, h) * 0.01
        self.W2 = np.random.randn(h, d_out) * 0.01
        self.cache = {}
           
    def forward(self, X):
        # TODO: Forward pass with ReLU activation
        Z1 = X @ self.W1
        A1 = np.maximum(0,Z1)
        logits = A1 @ self.W2
        self.cache['X'] = X
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1

        return logits
        
    
    def compute_loss(self, logits, y_true):
        # TODO: Softmax cross-entropy
        N = logits.shape[0]
        exp1 = np.exp(logits)
        soft_p = exp1 / np.sum(exp1,axis=1,keepdims=True)
        epsilon = 1e-8
        cross_entropy_sum = -np.sum(y_true * np.log(soft_p + epsilon))
        loss = cross_entropy_sum / N
        d_logits = (soft_p - y_true) / N
        return loss,d_logits
 
    
    def backward(self, d_logits):
        # TODO: Backward
        X = self.cache['X']
        Z1 = self.cache['Z1']
        A1 = self.cache['A1']
        dW2 = A1.T @ d_logits
        dA1 = d_logits @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        return {'dW1': dW1, 'dW2': dW2}

# Example usage
np.random.seed(0)
d_in, h, d_out, N = 3, 4, 2, 5
X = np.random.randn(N, d_in)
y = np.eye(d_out)[np.random.randint(0, d_out, N)]

model = TwoLayerMLP(d_in, h, d_out)
logits = model.forward(X)
loss, d_logits = model.compute_loss(logits, y)
grads = model.backward(d_logits)

print("dW1:\\n", grads['dW1'].round(4))
print("dW2:\\n", grads['dW2'].round(4))
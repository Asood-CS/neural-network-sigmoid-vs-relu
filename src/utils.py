#Importing neural network modules from PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, torch
from torch.func import functional_call
from torch.autograd.functional import hvp

class SimpleNeuralNetwork(nn.Module):
  #Initializes a standard neural network with one hidden layer
  def __init__(self, activation = torch.relu):
    #Hidden layer size -> tunable hyperparameter
    hidden_size = 32
    super().__init__()
    self.fc1 = nn.Linear(11, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    #One neuron in output layer -> binary classifier
    self.output = nn.Linear(hidden_size, 1)
    self.activation = activation

  def forward(self, x):
    #Forward propagation - currently running on the ReLU activation function
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.output(x)
    x = torch.sigmoid(x)
    return x

model = SimpleNeuralNetwork()
#Using the Binary Cross-Entropy loss function, a Maximum Likelihood Estimator (MLE) for binary classification
criterion = nn.BCELoss()
#Adam optimizer yields general network performance gains
optimizer = optim.Adam(model.parameters(), lr=5e-4)

#Calculating spectral norm of weight matrix
def spectral_norm(layer):
  weight = layer.weight.data
  #Conducting singular value decomposition on weight matrices
  U, S, V = torch.linalg.svd(weight, full_matrices=False)
  lip = S.max()
  return lip

#Finding Lipschitz constant of model outputs w.r.t inputs
def lipschitz_upper_bound(model, activation_bound):
  upper_bound = 1.0
  num_layers = len(list(model.modules()))
  #Iterating over layers to find data points
  for i, layer in enumerate(model.modules()):
    if isinstance(layer, nn.Linear):
      #Scaling spectral norm by activation function bounds
      norm = spectral_norm(layer)
      upper_bound *= norm
      if i < num_layers - 1:
        upper_bound *= activation_bound
      else:
        upper_bound *= 0.25
  return upper_bound

#Finding the Lipschitz constant w.r.t model gradients (Jacobian matrix)
def lipschitz_lower_bound(model, x):
  #Formatting inputs and outputs for gradient calculations
  x = x.clone().detach().requires_grad_(True)
  y = model(x)
  y = y.squeeze(-1)
  #Finding the final Jacobian matrix, differentiating loss w.r.t outputs
  grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
  #Using the L2 norm operation to find Euclidean distance
  grad_norm = torch.norm(grad, p=2, dim=-1)
  return grad_norm.max().detach().numpy()

def max_hessian_eigval(model, criterion, x, y, iters=20, tol=1e-6):

    #Creating a tensor of trainable parameters
    p_items      = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names        = [n for n, _ in p_items]
    params0      = tuple(p.detach().clone().requires_grad_(True) for _, p in p_items)
    buffers_dict = dict(model.named_buffers())

    #Formatting loss function to accept parameter tensor
    def loss_fn(*params):
        state = {**{k: v for k, v in zip(names, params)}, **buffers_dict}
        out   = functional_call(model, state, (x,))
        return criterion(out, y)

    #Defining tensor operations to compute Hessian-vector products (HVPs)
    def tup_dot(a, b):  
        return sum((ai.flatten() * bi.flatten()).sum() for ai, bi in zip(a, b))
    def tup_norm(a):   
        return math.sqrt(tup_dot(a, a).item())
    def tup_scale(a, s):
        return tuple(ai / s for ai in a)

    #Power iteration algorithm -> calculating repeated HVPs
    v = tuple(torch.randn_like(p) for p in params0) 
    v = tup_scale(v, tup_norm(v))                  

    for _ in range(iters):
        _, Hv = hvp(loss_fn, params0, v)            
        nrm    = tup_norm(Hv)
        if nrm < tol:                                
            return 0.0
        v = tup_scale(Hv, nrm)

    #Calculating the Rayleigh Quotient to isolate largest eigenvalue
    _, Hv = hvp(loss_fn, params0, v)
    return tup_dot(v, Hv).item()

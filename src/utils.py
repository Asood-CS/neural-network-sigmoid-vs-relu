#Master code file for my neural networks project

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, torch
from torch.func import functional_call
from torch.autograd.functional import hvp
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

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


#Fetching dataset from open-source machine learning repo
wine_quality = fetch_ucirepo(id=186)

#Turning the data into features and outputs
X = wine_quality.data.features
y = wine_quality.data.targets

#Organizing features as a Pandas dataframe
df_features = wine_quality.data.features
df_features.hist(column='alcohol', bins=50)

#Splitting output features into two classes
targets = wine_quality.data.targets
labels = (targets >= 7).astype(int)
labels.value_counts()
#Transforming labels to a Numpy-editable format
labels=labels["quality"].to_numpy()
X = df_features.to_numpy()

#Splitting datasets into training, validation, and test groups
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size=0.20, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)
x_train.shape

#Z-score scaling
m = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train-m)/std

x_val = (x_val-m)/std
x_test = (X_test-m)/std

#Creating the trainable data set in randomly shuffled batches
num_epochs = 75
dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train[:, None], dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val[:, None], dtype = torch.float32))
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def calculate_f1(outputs, labels, threshold):
  #Finding accurate model predictions
  predictions = (outputs >= threshold).int()
  #Creating a confusion matrix of predictions
  true_positives = ((predictions == 1) & (labels == 1)).sum().item()
  false_positives = ((predictions == 1) & (labels == 0)).sum().item()
  false_negatives = ((predictions == 0) & (labels == 1)).sum().item()
  #Balancing precision and recall using the F1 Score algorithm
  precision = true_positives / (true_positives + false_positives + 1e-8)
  recall = true_positives / (true_positives + false_negatives + 1e-8)
  f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
  return f1

def find_best_f1(outputs, labels):
  best_f1 = 0
  best_threshold = 0
  #Identifying which model threshold yields the best F1 Score
  for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    f1 = calculate_f1(outputs, labels, threshold)
    if f1 > best_f1:
      best_f1 = f1
      best_threshold = threshold
  return best_f1, best_threshold

columns = [
    'epoch',             # int or NaN if unknown
    'step',              # int or NaN if unknown
    'activation',        # string ('relu' or 'sigmoid')
    'train_loss',        # float (NaN if not logged at step)
    'val_loss',          # float (NaN if not logged at step)
    'f1_score',          # float (NaN if not logged at step)
    'lipschitz_upper',   # float (NaN if not logged at epoch)
    'lipschitz_lower',   # float (NaN if not logged at epoch)
    'max_hessian_eigval' # float (NaN if not logged at epoch)
]

#Transforming these parameters into a data frame
df = pd.DataFrame({col: pd.Series(dtype='float') for col in columns})

# Explicitly setting the activation column data type as string
df['activation'] = df['activation'].astype('string')

def train_and_log(activation_name, activation_fn):
    #Re-establishing key hyperparameters
    factor = 0.95
    model = SimpleNeuralNetwork(activation=activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCELoss()

    logs = []
    train_loss = None

    #Training over multiple epochs
    for epoch in range(num_epochs):
        model.train()
        for step, (batch_x, batch_y) in enumerate(dataloader):
            #Calculating Lipschitz metrics at the start of each training step
            optimizer.zero_grad()

            lower = lipschitz_lower_bound(model, batch_x)
            upper = lipschitz_upper_bound(model, 0.25)

            #Calculating loss and performing backpropagation
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            optimizer.step()

            #Calculating largest Hessian eigenvalue
            lam_max = max_hessian_eigval(model, criterion, batch_x.detach(), batch_y.detach())

            if train_loss is None:
                train_loss = loss.detach().item()
            else:
                train_loss = factor * train_loss + (1 - factor) * loss.detach().item()

            #Logging metrics per step on data frame
            logs.append({
                'epoch': epoch,
                'step': step,
                'activation': activation_name,
                'train_loss': np.nan,
                'val_loss': np.nan,
                'f1_score': np.nan,
                'lipschitz_upper': upper,
                'lipschitz_lower': lower,
                'max_hessian_eigval': lam_max
            })

        #Running model in evaluation mode against validation data
        model.eval()
        with torch.no_grad():
            val_labels = []
            val_predictions = []
            val_outputs = []
            val_losses = []

            #Calculating loss metrics for the validation set 
            for val_x, val_y in val_dataloader:
                outputs = model(val_x)
                loss = criterion(outputs, val_y)
                predictions = (outputs >= 0.5).int()
                val_labels.append(val_y)
                val_predictions.append(predictions)
                val_outputs.append(outputs)
                val_losses.append(loss)

            val_labels = torch.cat(val_labels, dim=0)
            val_predictions = torch.cat(val_predictions, dim=0)
            val_outputs = torch.cat(val_outputs, dim=0)
            val_loss = torch.stack(val_losses).mean().item()

            best_f1, best_threshold = find_best_f1(val_outputs, val_labels)

        #Condensing step-level logs into an epoch-level log
        logs.append({
            'epoch': epoch,
            'step': np.nan,
            'activation': activation_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'f1_score': best_f1,
            'lipschitz_upper': np.nan,
            'lipschitz_lower': np.nan,
            'max_hessian_eigval': np.nan
        })
    return logs

#Training and logging for both activation functions
all_logs = []

for act_name, act_fn in [('relu', F.relu), ('sigmoid', torch.sigmoid)]:
    logs = train_and_log(act_name, act_fn)
    all_logs.extend(logs)

#Creating a combined data frame after all runs
df_all = pd.DataFrame(all_logs)


#Importing neural network modules from PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import plotly.graph_objects as go

# Generate the Dataset
def generate_data(n_samples=1000):
  X = torch.zeros(n_samples, 2)
  y = torch.zeros(n_samples, dtype=torch.long)

  # Generate samples from two Gaussian distributions
  X[:n_samples//2] = torch.randn(n_samples//2, 2) + torch.Tensor([3,2])
  X[n_samples//2:] = torch.randn(n_samples//2, 2) + torch.Tensor([-3,2])

  # Labels
  for i in range(X.shape[0]):
    if X[i].norm() > math.sqrt(13):
      y[i] = 1

  X[:, 1] = X[:, 1] - 2

  return X, y

data, labels = generate_data()

class_0 = data[labels == 0]
class_1 = data[labels == 1]

# Scatter plots for Class 0 and Class 1
scatter_class_0 = go.Scatter(
    x=class_0[:, 0],
    y=class_0[:, 1],
    mode='markers',
    marker=dict(color='rgba(152, 0, 0, 0.5)'),
    name='Class 0'
)

scatter_class_1 = go.Scatter(
    x=class_1[:, 0],
    y=class_1[:, 1],
    mode='markers',
    marker=dict(color='rgba(0, 152, 0, 0.5)'),
    name='Class 1'
)

layout = go.Layout(
    xaxis=dict(title='Feature 1'),
    yaxis=dict(title='Feature 2'),
    showlegend=True,
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=0),
    hovermode='closest',
    plot_bgcolor='rgba(255, 255, 255, 0.8)'
)

fig = go.Figure(data=[scatter_class_0, scatter_class_1], layout=layout)
fig.write_image("figures/output_3_0.png", scale = 8, width=600, height=400)

class Expert(nn.Module):
  def  __init__(self, input_size, output_size): 
    super(Expert, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    
  def forward(self, data):
    x = self.linear(data)
    return x

class GatingNetwork(nn.Module):
  def __init__(self, input_size, num_experts):
    super(GatingNetwork, self).__init__()
    self.linear1 = nn.Linear(input_size, 4)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(4, num_experts)
    self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, data): 
    x = self.linear1(data)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.softmax(x)
    return x

class MixtureOfExperts(nn.Module):
  def __init__(self, num_experts=2):
    super(MixtureOfExperts, self).__init__()  
    self.expert1 = Expert(2,1)
    self.expert2 = Expert(2,1)
    self.gating =  GatingNetwork(2, num_experts)
    self.sigmoid = nn.Sigmoid()
      
  def forward(self, data):
    expert1_output = self.expert1(data)
    expert2_output = self.expert2(data)  
    
    gating_output =  self.gating(data)

    s = (expert1_output*gating_output[:,0][:,None] + 
         expert2_output*gating_output[:,1][:,None])
    
    a = self.sigmoid(s)
    
    return a

  def backward(self, y_hat, labels, criterion, optimizer): 
    optimizer.zero_grad()
    loss = criterion(y_hat, labels)    
    loss.backward()
    optimizer.step()
    return loss.item()

# Define the model, loss, and optimizer
moe = MixtureOfExperts()
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(moe.parameters(),lr=0.01)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer) 

# Convert data and labels to float tensors
data_tensor = data.float()
labels_tensor = labels.view(-1, 1).float()

# Training loop
num_epochs = 500
for epoch in tqdm(range(num_epochs)):
    # Forward pass
    y_hat = moe.forward(data)

    # Backward pass and optimization
    loss_value = moe.backward(y_hat, labels_tensor, criterion, optimizer)

    # Decay the learning rate
    scheduler.step()

print(sum(torch.round(y_hat).squeeze() == labels)/1000)

expert1_weights = moe.expert1.linear.weight.detach()[0,0]
expert2_weights = moe.expert2.linear.weight.detach()[0,0]

expert1_bias = moe.expert1.linear.bias.detach()
expert2_bias = moe.expert2.linear.bias.detach()

gating_weights = moe.gating.linear2.weight.detach().flatten()

x_line = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)

y_line1 = expert1_weights * x_line + 5
y_line2 = expert2_weights * x_line + 5

class_0 = data[labels == 0]
class_1 = data[labels == 1]

scatter_class_0 = go.Scatter(
    x=class_0[:, 0],
    y=class_0[:, 1],
    mode='markers',
    marker=dict(color='rgba(152, 0, 0, 0.5)'),
    name='Class 0'
)

scatter_class_1 = go.Scatter(
    x=class_1[:, 0],
    y=class_1[:, 1],
    mode='markers',
    marker=dict(color='rgba(0, 152, 0, 0.5)'),
    name='Class 1'
)

# Line plots for Expert 1 and Expert 2
line_expert_1 = go.Scatter(
    x=x_line,
    y=y_line1,
    mode='lines',
    line=dict(color='rgba(152, 0, 0, 0.5)'),
    name='Expert 1'
)

line_expert_2 = go.Scatter(
    x=x_line,
    y=y_line2,
    mode='lines',
    line=dict(color='rgba(0, 152, 0, 0.5)'),
    name='Expert 2'
)

layout = go.Layout(
    xaxis=dict(title='Feature 1'),
    yaxis=dict(title='Feature 2'),
    showlegend=True,
    autosize=True,
    margin=dict(l=0, r=0, b=0, t=0),
    hovermode='closest',
    plot_bgcolor='rgba(255, 255, 255, 0.8)'
)

fig = go.Figure(data=[scatter_class_0, scatter_class_1, line_expert_1, line_expert_2], layout=layout)

fig.write_image("figures/output_15_0.png", scale = 12, width=600, height=300)

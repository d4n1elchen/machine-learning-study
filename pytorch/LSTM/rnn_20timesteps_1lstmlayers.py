# Adapted from: https://github.com/thundercomb/pytorch-stock-predictor-rnn/blob/master/pytorch-stock-predictor-rnn.py

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Globals
INPUT_SIZE = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# Hyper parameters
learning_rate = 0.001
num_epochs = 100

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with INPUT_SIZE timesteps and t+1 output
X_train = []
y_train = []
for i in range(INPUT_SIZE, 1258):
    X_train.append(training_set_scaled[i-INPUT_SIZE:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))



# Importing the PyTorch libraries and packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Define StockRegressor
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h):
        y, h = self.rnn(x, h)
        out = self.linear(y.view(-1, self.hidden_size))
        return out, h

# Initialising the RNN
model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# Initialising optimiser
optimiser = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training
hidden_state = None

for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(X_train).float())
    labels = Variable(torch.from_numpy(y_train).float())

    output, hidden_state = model(inputs, hidden_state)

    loss = criterion(output.view(-1), labels)
    optimiser.zero_grad()
    loss.backward(retain_graph=True) # back propagation
    optimiser.step()                 # update the parameters
    
    print('epoch {}, loss {}'.format(epoch,loss.item()))




# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values
real_stock_price = np.concatenate((training_set[0:1258], test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_stock_price = sc.fit_transform(real_stock_price)
X_test = []
for i in range(1258, 1278):
    X_test.append(scaled_real_stock_price[i-INPUT_SIZE:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

inputs = np.concatenate((X_train, X_test),axis=0)
inputs = Variable(torch.from_numpy(inputs).float())
hidden_state = None
predicted_stock_price, hidden_state = model(inputs, hidden_state)
predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (inputs.shape[0], 1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# Adapted from: https://github.com/thundercomb/pytorch-stock-predictor-rnn/blob/master/pytorch-stock-predictor-rnn.py
#%%
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Globals
LOOKBACK_SIZE = 20
INPUT_SIZE = 2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# Hyper parameters
learning_rate = 0.001
num_epochs = 100

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv',thousands=',')
training_set = dataset_train.iloc[:,1:5].values

#%%
# Visualising the training set
plt.plot(training_set[:, 0], color = 'red', label = 'Real Google Stock Price')
plt.title('Google Stock Price Training data')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Feature Scaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler(feature_range = (0, 1))
sc_y = MinMaxScaler(feature_range = (0, 1))
# Scaling
training_set_scaled_X = sc_X.fit_transform(training_set[:, 1:3])
training_set_scaled_y = sc_y.fit_transform(training_set[:, 0:1])

# Creating a data structure with LOOKBACK_SIZE timesteps and t+1 output
X_train = []
y_train = []
for i in range(LOOKBACK_SIZE, 1258):
    X_train.append(training_set_scaled_X[i-LOOKBACK_SIZE:i])
    y_train.append(training_set_scaled_y[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.swapaxes(X_train, 0, 1)


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

    for i in range(LOOKBACK_SIZE):
        output, hidden_state = model(inputs[i, :, :].view(1, -1, INPUT_SIZE), hidden_state)

    loss = criterion(output, labels)
    optimiser.zero_grad()
    loss.backward(retain_graph=True) # back propagation
    optimiser.step()                 # update the parameters
    
    print('epoch {}, loss {}'.format(epoch,loss.item()))




# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv',thousands=',')
test_set = dataset_test.iloc[:,1:5].values
real_stock_price = np.concatenate((training_set[0:1258], test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_stock_price_X = sc_X.fit_transform(real_stock_price[:, 1:3])

X_test = []
for i in range(1258, 1278):
    X_test.append(scaled_real_stock_price_X[i-LOOKBACK_SIZE:i])
X_test = np.array(X_test)
X_test = np.swapaxes(X_test, 0, 1)

inputs = np.concatenate((X_train, X_test),axis=1)
inputs = Variable(torch.from_numpy(inputs).float())
hidden_state = None
for i in range(LOOKBACK_SIZE):
    predicted_stock_price, hidden_state = model(inputs[i, :, :].view(1, -1, INPUT_SIZE), hidden_state)
predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (-1, 1))
predicted_stock_price = sc_y.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[:, 0], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# from google.colab import files
# uploaded = files.upload()

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data_NN.csv",sep=";",header=0,names=("d1","d2","d3"),decimal=",")
X_data = np.concatenate((np.array([[-2,0,-1]]),data[:-1]), axis=0)
y_data = np.array(data)

n_features = np.size(X_data,1)
n_outputs = np.size(y_data,1)

x = torch.from_numpy(X_data)
y = torch.from_numpy(y_data)

# Defining input size, input layer size, hidden layer size, output size and batch size respectively
input_size, n_in, n_h, n_out, batch_size = n_features, 3, 3, n_outputs, 1000

# Create a model
model = nn.Sequential(
    nn.Linear(input_size,n_in),
    nn.Tanhshrink(),
    nn.Linear(n_in,n_h),
    nn.Tanhshrink(),
    nn.Linear(n_h,n_out),
    nn.Tanhshrink()).double()

# Construct the loss function
criterion = torch.nn.L1Loss()

# Construct the optimizer (Adamax in this case)
optimizer = torch.optim.Adamax(model.parameters(), lr = 0.0005, weight_decay=0.0005) 

# Gradient Descent
epochs = 100000
losses = []
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    prediction = model(x)

    # Compute and print loss
    error = criterion(prediction, y)
    y_pred, loss = prediction, error
    losses.append(loss.item())
    if epoch % (epochs/10) == (epochs/10-1):
        print('epoch: ', epoch,' loss: ', loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()


# visualizing the error after each epoch
cut = int(epoch*0.1)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Error on training data after each epoch')
ax1.plot(np.arange(1, cut+1), np.array(losses[:cut]))
ax1.set_ylabel('epochs 1 to %i' %(cut+1))
ax2.plot(np.arange(cut+1, epoch+1), np.array(losses[cut+1:]))
ax2.set_xlabel('epochs')
ax2.set_ylabel('epochs '+str(cut+1)+' to '+str(epoch))
plt.show()
print(np.array(losses).min(), losses[epoch])
y_predicted = y_pred.detach().numpy()


# 3D plotting
def d3plot(data, label, colour, title='3d plot'):
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111,projection='3d')
    if (type(data) is not tuple): data, label, colour = (data,), (label,), (colour,)
    for i in range(len(data)):
        ax.plot3D((data[i])[:,0], (data[i])[:,1], (data[i])[:,2], colour[i], label=label[i])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    return plt.show()


d3plot((y_predicted, y_data), label=('predicted','f(Xκ)'), colour=('r','b'), 
       title='Η απόκριση του συστήματος και η εκτίμησή της από το νευρωνικό για τα δεδομένα Xκ')

xk = torch.tensor([[-1.9, 0.0, -0.9]], dtype=torch.double)
x_val = []
steps = 200
for i in range(steps):
    xk = model(xk)
    x_val.append(xk.detach().numpy())

x_val = np.squeeze(x_val)

d3plot((x_val, y_data[:steps]), label=('x_val','f(Xκ)'), colour=('g','b'), 
        title='Γραφική απεικόνιση της πρόβλεψης του νευρωνικού για την απόκριση του συστήματος'
        '\n με αρχική κατάσταση x0=(-1.9, 0, -0.9)  για 200 επαναλήψεις,'
        '\n σε αντιπαραβολή με τις 200 πρώτες μετρήσεις για την απόκριση του συστήματος'
        '\n με αρχική κατάσταση x0=(-2, 0, -1)')

x_val = (np.squeeze(x_val)).T


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #


# # Construct the optimizer (Adamax in this case)
# optimizer = torch.optim.Adamax(model.parameters(), lr = 0.00001, weight_decay=0.00001)

# # Gradient Descent
# epochs = 100
# for epoch in range(epochs):
#     # Forward pass: Compute predicted y by passing x to the model
#     xk = x[0,:]
#     matrix = torch.zeros(1000, 3, dtype=torch.double)
#     for i in range(1000):
#         xk = model(xk)
#         matrix[i,:] = xk
    
#     y_pred = matrix
#     # Compute and print loss
#     loss = criterion(y_pred, y)

# # visualizing the error after each epoch

# xk = torch.tensor([[-1.9, 0.0, -0.9]], dtype=torch.double)

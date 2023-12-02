import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return F.log_softmax(x, dim=1)
    
net = Net()
print(net)

# as long as our data is already scaled between 0 to 1 and is equilibred we just have to iterate throught it
# to train our model calculatingthe loss and specifying the optimicer

import torch.optim as optim

# calculate how far of our results are from reality
loss_funtion = nn.CrossEntropyLoss()

# If the desired result were a one-hot matrix, we would use mean squared error, but since we want the resulting value 
# to be a scalar, we use CrossEntropy as the error measure. Ex: one hot matrix for 3 in our case could be -> [0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0]

# Adjust the configurable parameters of our model Adam = adaptative momentum
optimizer = optim.Adam(net.parameters(), lr=0.001)

# epoch = Complete pass through the training dataset. few epoch = model dont learn the necessary. To much epoch = overfitting

for epoch in range(3): # 3 full passes over the data
    for data in trainset: # 'data' is a batch of data (10 images in our  case)
        X,y = data # X is the batch of features, y is the batch of targets
        net.zero_grad() # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1, 28*28)) # pass in the reshaped batch (recall they are 28x28 atm)
        loss = F.nll_loss(output,y) # calc and grab the loss value
        loss.backward() # apply this loss backwards thru the network's parameters
        optimizer.step() # attempt to optimize weights to account for loss/gradients
    print(loss) # # print loss. We hope loss (a measure of wrong-ness) declines! 

# get the accuracy
correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X,y = data
        output = net(X.view(-1,784))
        # print(output)
        for idx, i in enumerate(output):
            # print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct +=1
            total +=1

print("Accuracy: ", round(correct/total,3))

import matplotlib.pyplot as plt

plt.imshow(X[0].view(28,28))
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))

# same as before but explained:
# a_featureset = X[0]
# reshaped_for_network = a_featureset.view(-1,784) # 784 b/c 28*28 image resolution.
# output = net(reshaped_for_network) #output will be a list of network predictions.
# first_pred = output[0]
# print(first_pred)
# Which index value is the greatest? We use argmax to find this:
# biggest_index = torch.argmax(first_pred)
# print(biggest_index)
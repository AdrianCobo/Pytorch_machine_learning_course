import torch
import torchvision
from torchvision import transforms, datasets 

# loading data, Shuffling and applying transformation or preprocessing
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

# deciding how we are going to iterate through the data
trainset = torch.utils.data.DataLoader(train,batch_size=10, shuffle = True)
trainset = torch.utils.data.DataLoader(test,batch_size=10, shuffle = False)

import torch.nn as nn # neural networks
import torch.nn.functional as F # activation functions

# creat our model as a class
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__() # initialize parent class

# net = Net()
# print(net)

# creat our model as a class
class Net(nn.Module):
    def __init__(self):
        super().__init__() # initialize parent class
                         #(input, output) size. our images ar 28x28 pixels size
        self.fc1 = nn.Linear(28*28, 64) # fc = standard name for fully connected
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64) # 64 = randomly chosed
        self.fc4 = nn.Linear(64,10) # output 10 because we have 10 classes

    # we are going to do a fully connected feedforward network
    # def forward(self,x): # x = input data
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #     x = self.fc4(x)
    #     return x
    # feed forward network fully connected with relu activation function for hidden layers(scale data between 0-1)
    # and softmax as activation funcion on the last layer because the result is going to be a confiddence score, adding
    # up to 1 (usefull for multicass problems)
    # *activation function is used in orther to see if the neurons are being activated or not
    def forward(self,x): # x = input data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x),dim=1)
        return x
        

net = Net()
print(net)

# lets see if our neural network work
X = torch.randn((28,28))
X = X.view(-1,28*28) # flattened data 28x28 to 1x(28*28). -1 means any value of rows (the variable part is how many "samples" we'll pass through)
print(net(X)) # tensor means variable list (more or less)
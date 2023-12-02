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

# iterate through the dataset: we are going to have 10 elements per batch(because we chose it) and 10 classes
for data in trainset:
    print(data)
    break


X,y = data[0][0], data[1][0] # x = Input -> data[0], y = output -> data[1]

import matplotlib.pyplot as plt # pip install matplotlib

plt.imshow(data[0][0].view(28,28))
plt.show()

# lets see if our data is scaled between 0-1
print(data[0][0][0][0]) # fully 0 row
print(data[0][0][0][9]) # yep, de data is between 0-1

# Is our dataset balanced? same amout of data of each class
total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] +=1
        total +=1

print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100.0}%")

# yep, the dataset is quite balanced
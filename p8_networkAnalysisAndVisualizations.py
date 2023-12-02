# https://pythonprogramming.net/convolutional-neural-networks-deep-learning-neural-network-pytorch/
# dataset: https://www.kaggle.com/datasets/tongpython/cat-and-dog/

# Convolution -> Create a sliding window that will compare features in data clusters.
# Pooling -> Generate a "new data" using a sliding window that will select the maximum value, the average, etc., of those features.
# No GPU
# pip install opencv-python numpy tqdm matplotlib
# With GPU and my cuda version:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm

# set to true to one once, then back to false unless you want to change something in your training data
REBUILD_DATA = False
# print(torch.cuda.is_available()) # see if Cuda Pytorch is available
# print(torch.cuda.device_count()) # see how many GPUs are available on your sistem. You can assign specific layers to specific GPUs

# data processing class


class DogsVSCats():
    IMG_SIZE = 50  # we are going to reshape the images to 50x50

    # directories of the data
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    TESTING = "PetImages/Testing"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    # dataset balance counter variables
    catcount = 0
    dogcount = 0

    # We want to iterate through these two directories, grab the images, resize, scale, convert the class to number (cats = 0, dogs = 1),
    # and add them to our training_data.

    # All we're doing so far is iterating through the cats and dogs directories, and looking through all of the images
    # and handle for the images:

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # just makes one_hot matrix as targets.
                        self.training_data.append(
                            [np.array(img), np.eye(2)[self.LABELS[label]]])
                        # example of np.eye(2)[1] -> [0. 1.]

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass
                        # print(label, f, str(e))

        print(self.training_data[0])
        np.random.shuffle(self.training_data)
        # dtype because [image,result] have differents sizes
        np.save("training_data.npy", np.array(
            self.training_data, dtype=object))
        print('Cats:', dogsvcats.catcount)
        print('Dogs:', dogsvcats.dogcount)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

# get the data
training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))


# data -> convolutional layer -> fully connected layer -> output layer


class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        # run the init from the parent class (nn.Module)
        super().__init__(*args, **kwargs)
        # input is 1 image, 32 output channels, 5x5 kernel/window
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 32, bc = the first layer output 32. Then we say output will be 64 channels, 5x5 conv
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # basically we take 1 image, and generate 32 convolutions/caracteristics. The next layer will take the 32 characteristics ang generate 64,...
        # the convolutions are not flat, so we need to flatten them(like with the images) before passing them to the next layer. In order to know
        # what is the size of the convolutions that we need to flat, we are going to pass "fake data" one time to the Net and see it.

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        # 512 in, 2 out bc we'are doing classes (dog vs cat)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # max poooling over 2x2
        # window of 2x2 for pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            # calculate the dimension of the tensor
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
            # print(self._to_linear) # -> 512
        return x

    def forward(self, x):
        x = self.convs(x)
        # .view is reshape ... this flatens X before
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here
        return F.softmax(x, dim=1)


net = Net()
# device = torch.device("cuda:0")
# print(device)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# set our reural network to our device (you can assing diferent neuran network layers to differents GPUs if you have them)
print(net.to(device))
# net = Net().to(device) # assing a new net to our GPU

# you can move all the data to the GPU because in this case its not to much big but you normaly won't and what you do is move the batch data.

# training and optimize: optimizer is going to be Adam and because we are using one hot matix we use the MSE error metric

# split the data into X and y and convert it into a tensor


optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

# separate some data into training and 10% for testing
VAL_PCT = 0.1  # lest reserve 10% of our data for validation
# converting it to int because we are goint to slice our data in groups of it so it needs to be a valid index
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X), len(test_X))

import time

MODEL_NAME = f"model-{int(time.time())}" # gives a dynamic model name to just help with things getting messy over time. 

# iterate throught the data using batches and calculate train accuracy

def fwd_pass(X,y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1,1,50,50).to(device), y.to(device),train=False)
    return val_acc, val_loss

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 30

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            # from 0, to the len of x, stepping BACH_SIZE at a time.
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                # print(f"{i}:{i+BATCH_SIZE}")
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)} Loss: {round(float(loss),4)}")
                # analice training acc and loss and test acc and loss each 10 steps
                if i % 10 == 0:
                    val_acc, val_loss = test(size=BATCH_SIZE)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")
train(net)

import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

    ax1.plot(times, accuracies, label="train_acc")
    ax1.plot(times, val_accs, label="test_acc")
    ax1.legend(loc=2)
    ax2.plot(times,losses, label="train_loss")
    ax2.plot(times, val_losses, label="test_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(MODEL_NAME)

# each epoch is more or less 3 seconds
# you see the x axes at the plot, see for divergence for example x = 433 means timestamp = +value_x_idicated + x and see at what epoch it correspond.
# in our case at epoch 6-8 things start going wrong. and we should stop much before (epoch 3 or 4)

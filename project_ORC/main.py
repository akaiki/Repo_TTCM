import torch
import matplotlib.pyplot as plt 
import numpy as np
import torchvision

from torch import nn, optim
from torchvision import datasets, transforms

batch_size = 32 #lượng ảnh dùng trong 1 batch
learning_rate = 0.01 #tỉ lệ học được
num_epochs = 20 #số lần sử dụng toàn bộ dữ liệu để train

#dataloader
#load package dataset.MNIST có trong torchvision để tải về bộ dữ liệu vào mục data

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download =True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.137, ), (0.3081, ))
                  ])  
    ),
    batch_size=batch_size,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.137, ), (0.3081, )),
    ])),
    batch_size=batch_size,
    shuffle=False
)


#visualize data
def imshow(img, mean, std):
    img = img / std + mean # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images), 0.1307, 0.3081)
print(labels)

#get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model
from model import SimpleModel
model = SimpleModel().to(device)

# loss function
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#training
num_steps = len(train_loader)

for epoch in range(num_epochs):
    #----training----
    #set model to training

    model.train()

    total_loss = 0

    for i, (images,labels) in enumerate(train_loader):
        images,labels = images.to(device), labels.to(device)

        #zero gradients
        optimizer.zero_grad()

        #forward
        outputs = model(images)

        #compute loss
        loss = criterion(outputs, labels)

        #backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #print log
        if (i+1)&100 ==0:
            print("epoch {}/{} - step: {}/{} - loss: {:.4f}".format(
                epoch, num_epochs, i, num_steps, total_loss/(i+1)
            ))
    #----Validation----
    #set model to evaluating
    model.eval()

    val_losses = 0

    with torch.no_grad():
        correct =0
        total =0
        for _, (images,labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            loss = criterion(outputs,labels)

            val_losses += loss.item()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("epoch {} - accuracy: {} - validation loss: {:.4f}".format(
            epoch, correct / total, val_losses / (len(val_loader))
        ))


    
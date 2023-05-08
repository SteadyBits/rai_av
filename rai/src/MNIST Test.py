#!/usr/bin/env python
# coding: utf-8

import torch
import matplotlib.pyplot as plt
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

# plt.imshow(train_data.data[0], cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()


# figure = plt.figure(figsize=(10, 8))
# cols, rows = 5, 5
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

#Dataloader
BATCH_SIZE = 100
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders

#Model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

model = CNN()

#Loss function
loss_func = nn.CrossEntropyLoss()   

#Optimization function
from torch import optim
optimizer = optim.Adam(model.parameters(), lr = 0.01)   
optimizer

# # Training with 50,000 samples. Epoch = 10. Energy ~ 0.002169 kWh

#Training process OPTION 1
from torch.autograd import Variable

num_epochs = 10
def train(model, loaders, num_epochs):
    
    model.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            image_x = Variable(images)   # batch x
            image_y = Variable(labels)   # batch y
            output = model(image_x)[0]               
            loss = loss_func(output, image_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
        
        pass
    
    pass

train(model, loaders, num_epochs)

# # # 10,000 Predictions. Energy ~ 0.000030 KWh
# def test(model):
#     # Test the model
#     correct = 0
#     BATCH_SIZE = 100
#     model.eval()
    
#     with torch.no_grad():
#         for images, labels in loaders['test']:
#             test_output, last_layer = model(images)
#             pred_y = torch.max(test_output, 1)[1].data.squeeze()
#             correct += (pred_y == labels).sum().item()
            
#     print("Test accuracy:{:.3f}% ".format( float(correct) / (len(loaders['test'])*BATCH_SIZE)))

# test(model)

# Save the model to a file
torch.save(model.state_dict(), 'MNIST_Checkpoint.pth')
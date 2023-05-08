
#Model architecture
import time
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from core.responsibleAI import RAIModels

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

model_args = dict({'model_args': [
    {'sensor':'camera',
     'data': 0,
     'no_of_images': 1,
     'concat': True,
     'order': {'front': 0},
     'concat_axis': -1
     }
    ]})
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

# Load the model from the file
model.load_state_dict(torch.load('MNIST_Checkpoint.pth'))

model_accuracy = None
# # 10,000 Predictions. Energy ~ 0.000030 KWh
def test(model):
    # Test the model
    correct = 0
    BATCH_SIZE = 100
    model.eval()
    torch.set_grad_enabled(False)
    counter = 1

    #define input structure
    model_args = dict({'model_args': [
    {'sensor':'camera',
     'data': 0,
     'no_of_images': 1,
     'concat': True,
     'order': {'front': 0},
     'concat_axis': -1
     }
    ]})

    #create an instance of RAI class
    
    for images, labels in loaders['test']:
        #print(model_args)
        #print(images.shape)
        # print(images[0].shape)
        # a = np.concatenate((images[0], images[1], images[2]), axis=-1)
        # print(a.shape)
        model_args['model_args'][0]['data'] = images
        output = rAI_engine.predict(model_args)
        test_output, last_layer = output[0], output[1]
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        correct += (pred_y == labels).sum().item()
        print("Prediction... ", counter)
        counter += 1
        #if counter == 5:
        #    break
    model_accuracy = float(correct) / ((len(loaders['test']) * BATCH_SIZE))
    print("Test accuracy:{:.3f}% ".format(model_accuracy))
    return model_accuracy
    


iterations = len(model_args['model_args']) * model_args['model_args'][0]['no_of_images']
if 'noise_params' in model_args:
    iterations *= len(model_args['noise_params'])
else:
    iterations *= 4

rAI_engine = RAIModels(model, model_args)
print("iterlen+++++", iterations)
for i in range(iterations):
    model_accuracy = test(model)
    rAI_engine.register_model_rai(model_accuracy)

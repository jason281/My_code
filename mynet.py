import torch.nn as nn

__all__ = ['MyNet', 'mynet']

class MyNet(nn.Module):
    def __init__(self, classN = 10):
        super(MyNet, self).__init__()
        self.conv1    = nn.Conv2d( 3,  8, kernel_size=3, stride=1, padding=1)
        self.relu1    = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d( 8,  8, kernel_size=3, stride=1, padding=1)
        self.relu2    = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3    = nn.Conv2d( 8, 16, kernel_size=3, stride=1, padding=1)
        self.relu3    = nn.ReLU(inplace=True)
        self.conv4    = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu4    = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5    = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu5    = nn.ReLU(inplace=True)
        self.conv6    = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu6    = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1      = nn.Linear(512, 100)
        self.relu7    = nn.ReLU(inplace=True)
        self.fc2      = nn.Linear(100, classN)
        
    def forward(self, x):
        x = self.conv1(x)   
        x = self.relu1(x)   
        x = self.conv2(x)   
        x = self.relu2(x)   
        x = self.maxpool1(x)
        x = self.conv3(x)   
        x = self.relu3(x)   
        x = self.conv4(x)   
        x = self.relu4(x)   
        x = self.maxpool2(x)
        x = self.conv5(x)   
        x = self.relu5(x)   
        x = self.conv6(x)   
        x = self.relu6(x)   
        x = self.maxpool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)     
        x = self.relu7(x)   
        x = self.fc2(x)
        return x
    
def mynet(inplanes, classN = 10):
    return MyNet(classN)

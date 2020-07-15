# models.py
from torch import nn

#mô hình mạng dùng bên main.py
class SimpleModel(nn.Module):  #class kế thừa từ nn.Module
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()

        #layer conv
        self.conv1 = nn.Sequential(
                #Bước 1: Conv2D
                nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        #layer Fully connected
        self.fc = nn.Linear(14 * 14 * 32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
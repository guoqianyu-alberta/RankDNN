import torch.nn as nn



class LinearModule(nn.Module):
    def __init__(self,input_size,num_classes):
        super(LinearModule, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return x, y
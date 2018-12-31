import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
nclasses = 20 

class VGG_custom(nn.Module):
    def __init__(self,num_outputs = nclasses):

        super(VGG_custom, self).__init__()
        
        vgg_model = torchvision.models.vgg19(pretrained=True)
        
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:2])
        
        for param in self.Conv1.parameters():
            param.requires_grad = False
            
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[2:5]) 
        
        for param in self.Conv2.parameters():
            param.requires_grad = False
        
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[5:7])
        
        for param in self.Conv3.parameters():
                param.requires_grad = False

        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[7:37])
        
        self.Dense1 = nn.Sequential(nn.Linear(25088,2048),nn.ReLU(),torch.nn.Dropout(p=0.5))
        self.Dense2 = nn.Sequential(nn.Linear(2048,64),nn.ReLU(),torch.nn.Dropout(p=0.5))
        self.Classifier = nn.Sequential(nn.Linear(64,num_outputs))
        
        
    def forward(self, x):
        #Convolutional layers          
        output = self.Conv1(x)
        output = self.Conv2(output)
        output = self.Conv3(output)
        output = self.Conv4(output)
        
        #Flatten
        output = output.view(-1,25088)
        
        #Dense layers
        output = self.Dense1(output)
        output = self.Dense2(output)
        
        #Classifier
        output = self.Classifier(output)

        return output


# class ResNet_custom(nn.Module):
#     def __init__(self,num_outputs = nclasses):
#         super(ResNet_custom, self).__init__()
#         ResNet_model = torchvision.models.resnet152(pretrained = True)
        
#         self.Conv1 = nn.Sequential(*list(ResNet_model.children())[:5])
        
#         for param in self.Conv1.parameters():
#             param.requires_grad = False
        
#         self.Conv2 = nn.Sequential(*list(ResNet_model.children())[5:-1])
        
#         self.Dense1 = nn.Sequential(
#              nn.Dropout(p=0.5),
#              nn.Linear(in_features=2048, out_features=64, bias=True),
#              nn.ReLU())
                
#         self.Classifier = nn.Sequential(nn.Linear(64,num_outputs))
        
        
#     def forward(self, x):
#         #Convolutional layers
#         output = self.Conv1(x)
#         output = self.Conv2(output)
        
#         #Flatten
#         output = output.view(-1,2048)
        
#         #Dense layers
#         output = self.Dense1(output)
        
#         #Classifier
#         output = self.Classifier(output)

#         return output
    
    


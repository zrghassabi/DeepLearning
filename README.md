# DeepLearning

data Loading, Traning Model, Testing
https://github.com/LinkedInLearning/pytorch-essential-training-deep-learning-2706322/tree/main

#@title Import and transform for training data set
                    

#@title Defining transform for testing data set

Data batching

          from torchvision import transforms
          
          from torchvision.datasets import CIFAR10
          
          from torch.utils.data import DataLoader

          train_data_path = "./train/"
          
          test_data_path = "./test/"

          train_transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))])
          
          training_data = CIFAR10(train_data_path,
                               train=True,
                               download=True,
                               transform=train_transforms)
          
          test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))])
          
          test_data = CIFAR10(test_data_path,
                               train=False,
                               download=True,
                               transform=train_transforms)
          
          batch_size=16
          
          training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
          
          test_data_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)



Model development and training

#@title Import libraries and dataset
          
                 
          from torchvision import models
          
          from torch import optim
          
          import torch.nn as nn
          
          import torch.optim as optim
          
          import torch.nn.functional as F



#@title Define neural network, __init__ and forward functions

          class Net(nn.Module):
          
              def __init__(self):
              
                  super(Net, self).__init__()
                  
                  self.conv1 = nn.Conv2d(3, 6, 5)
                  
                  self.pool = nn.MaxPool2d(2, 2)
                  
                  self.conv2 = nn.Conv2d(6, 16, 5)
                  
                  self.fc1 = nn.Linear(16 * 5 * 5, 120)
                  
                  self.fc2 = nn.Linear(120, 84)
                  
                  self.fc3 = nn.Linear(84, 10)
          
              def forward(self, x):
              
                  x = self.pool(F.relu(self.conv1(x)))
                  
                  x = self.pool(F.relu(self.conv2(x)))
                  
                  x = x.view(-1, 16 * 5 * 5)
                  
                  x = F.relu(self.fc1(x))
                  
                  x = F.relu(self.fc2(x))
                  
                  x = self.fc3(x)
                  
                  return x

                  
  #@title Instantiate the Model

    net = Net()

    # define the Loss Function and Optimizer

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
     
   

#@title Train the network

      for epoch in range(10):  # loop over the dataset multiple times
  
          running_loss = 0.0
          
          for i, data in enumerate(trainloader, 0):
          
              # get the inputs; data is a tuple of [inputs, labels]
              
              inputs, labels = data
      
              # zero the parameter gradients
              
              optimizer.zero_grad()
      
              # forward + backward + optimize
              
              outputs = net(inputs)
              
              loss = criterion(outputs, labels)
              
              loss.backward()
              
              optimizer.step()
              
      
              # print statistics
              running_loss += loss.item()
              
              if i % 2000 == 1999:    # print every 2000 mini-batches
              
                  print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                  
              running_loss = 0.0
                  
      
          print('Finished Training')     



Validation and testing

@title Set the model to training mode and evaluation mode for validation

          for epoch in range(10):
              net.train()  # Set the model to training mode
              running_loss = 0.0
              for i, data in enumerate(trainloader, 0):
                  inputs, labels = data
                  optimizer.zero_grad()
                  outputs = net(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()
                  running_loss += loss.item()
          
              net.eval()  # Set the model to evaluation mode for validation
              validation_loss = 0.0
              correct = 0
              total = 0
              with torch.no_grad():
                  for data in testloader:
                      images, labels = data
                      outputs = net(images)
                      loss = criterion(outputs, labels)
                      validation_loss += loss.item()
                      _, predicted = torch.max(outputs.data, 1)
                      total += labels.size(0)
                      correct += (predicted == labels).sum().item()
          
              print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(trainloader)}, Validation Loss: {validation_loss / len(valloader)}, Validation Accuracy: {100 * correct / total}%')

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


https://www.kaggle.com/timoboz/data-science-cheat-sheets


Interview Questions: https://www.kaggle.com/timoboz/data-science-cheat-sheets

SuperCheetSheet for Deep Learning : https://www.kaggle.com/timoboz/data-science-cheat-sheets


https://media-exp2.licdn.com/dms/document/C4E1FAQFCUDAxHf_mYA/feedshare-document-pdf-analyzed/0/1657472777326?e=1658361600&v=beta&t=RB71o9MDqt43VpWBYAGKH9Cl5Br6rFVP1XocwGlM3Ic

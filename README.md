# DeepLearning

data Loading, Traning Model, Testing
https://github.com/LinkedInLearning/pytorch-essential-training-deep-learning-2706322/tree/main

#@title Import and transform for training data set

          from torchvision import transforms
          
          from torchvision.datasets import CIFAR10
          
          train_data_path = "./train/"
          
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

                     

#@title Defining transform for testing data set

                    test_data_path = "./test/"
                    
                    test_transforms = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(
                          (0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010))])
                    
                    test_data = CIFAR10(test_data_path,
                                         train=False,
                                         download=True,
                                         transform=train_transforms)
                    
                    print(test_data)



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





https://www.kaggle.com/timoboz/data-science-cheat-sheets


Interview Questions: https://www.kaggle.com/timoboz/data-science-cheat-sheets

SuperCheetSheet for Deep Learning : https://www.kaggle.com/timoboz/data-science-cheat-sheets


https://media-exp2.licdn.com/dms/document/C4E1FAQFCUDAxHf_mYA/feedshare-document-pdf-analyzed/0/1657472777326?e=1658361600&v=beta&t=RB71o9MDqt43VpWBYAGKH9Cl5Br6rFVP1XocwGlM3Ic

from datasetBase import ClientSet
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def makeCIFAR10Data(num_users):
  trans_CIFAR = transforms.Compose([transforms.ToTensor(),])
  
  CIFAR_train = datasets.CIFAR10('./data/CIFAR10/', train=True, download=True, transform=trans_CIFAR)
  CIFAR_test = datasets.CIFAR10('./data/CIFAR10/', train=False, download=True, transform=trans_CIFAR)
  CIFAR_TEST_SET = DataLoader(CIFAR_test, batch_size=10, shuffle=True)
  
  CIFAR_loader = DataLoader(CIFAR_train, batch_size=len(CIFAR_train)//num_users, shuffle=True)
  datasets = []
  users = 0
  for images, labels in CIFAR_loader:
    client_dataset = ClientSet(images, labels)
    datasets.append(DataLoader(client_dataset, batch_size = 100, shuffle=True))
    users += 1
    if(users >= num_users):
      return datasets, CIFAR_test
  return datasets, CIFAR_test

from datasetBase import ClientSet
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def makeMNISTData(num_users):
  trans_mnist = transforms.Compose([transforms.ToTensor(),])

  mnist_train = datasets.MNIST('../data/MNIST/', train=True, download=True, transform=trans_mnist)
  mnist_test = datasets.MNIST('../data/MNIST/', train=False, download=True, transform=trans_mnist)
  MNIST_TEST_SET = DataLoader(mnist_test, batch_size=100, shuffle=True)
  
  mnist_loader = DataLoader(mnist_train, batch_size=len(mnist_train)//num_users, shuffle=True)
  datasets = []
  users = 0
  for images, labels in mnist_loader:
    client_dataset = ClientSet(images, labels)
    datasets.append(DataLoader(client_dataset, batch_size = 100, shuffle=True))
    users += 1
    if(users >= num_users):
      return datasets, mnist_test
  return datasets, mnist_test

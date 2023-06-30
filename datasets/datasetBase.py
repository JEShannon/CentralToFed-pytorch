import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd

#All datasets ultimately are used as client sets, which is just a wrapper for a non-changing list.
#Clientset is defined here, but how a clientset is created from a base dataset is implementation specifc.

class ClientSet(Dataset):
  def __init__(self, data, labels):
    self.images = data
    self.labels = labels
  def __len__(self):
    return len(self.labels)
  def __getitem__(self, idx):
    return self.images[idx], self.labels[idx]

#### CIFAR-10

def makeCIFAR10Data(num_users):
  trans_CIFAR = transforms.Compose([transforms.ToTensor(),])
  
  CIFAR_train = datasets.CIFAR10('./data/CIFAR10/', train=True, download=True, transform=trans_CIFAR)
  CIFAR_test = datasets.CIFAR10('./data/CIFAR10/', train=False, download=True, transform=trans_CIFAR)
  CIFAR_TEST_SET = DataLoader(CIFAR_test, batch_size=10, shuffle=True)
  
  CIFAR_loader = DataLoader(CIFAR_train, batch_size=len(CIFAR_train)//num_users, shuffle=True)
  userDatasets = []
  users = 0
  for images, labels in CIFAR_loader:
    client_dataset = ClientSet(images, labels)
    userDatasets.append(DataLoader(client_dataset, batch_size = 100, shuffle=True))
    users += 1
    if(users >= num_users):
      return userDatasets, CIFAR_test
  return datasets, CIFAR_test

#### CIFAR-100

def makeCIFAR100Data(num_users):
  trans_CIFAR = transforms.Compose([transforms.ToTensor(),])
  
  CIFAR_train = datasets.CIFAR100('./data/CIFAR/', train=True, download=True, transform=trans_CIFAR)
  CIFAR_test = datasets.CIFAR100('./data/CIFAR/', train=False, download=True, transform=trans_CIFAR)
  CIFAR_TEST_SET = DataLoader(CIFAR_test, batch_size=10, shuffle=True)
  
  CIFAR_loader = DataLoader(CIFAR_train, batch_size=len(CIFAR_train)//num_users, shuffle=True)
  userDatasets = []
  users = 0
  for images, labels in CIFAR_loader:
    client_dataset = ClientSet(images, labels)
    userDatasets.append(DataLoader(client_dataset, batch_size = 100, shuffle=True))
    users += 1
    if(users >= num_users):
      return datasets, CIFAR_test
  return datasets, CIFAR_test

#### MNIST

def makeMNISTData(num_users):
  trans_mnist = transforms.Compose([transforms.ToTensor(),])

  mnist_train = datasets.MNIST('../data/MNIST/', train=True, download=True, transform=trans_mnist)
  mnist_test = datasets.MNIST('../data/MNIST/', train=False, download=True, transform=trans_mnist)
  MNIST_TEST_SET = DataLoader(mnist_test, batch_size=100, shuffle=True)
  
  mnist_loader = DataLoader(mnist_train, batch_size=len(mnist_train)//num_users, shuffle=True)
  userDatasets = []
  users = 0
  for images, labels in mnist_loader:
    client_dataset = ClientSet(images, labels)
    userDatasets.append(DataLoader(client_dataset, batch_size = 100, shuffle=True))
    users += 1
    if(users >= num_users):
      return userDatasets, mnist_test
  return userDatasets, mnist_test

#### Breast Cancer Wisconsin

#this dataset is already in our folder, under "BreastCancerWisconsin/wdbc.csv"
def makeBCWData(num_users):
  BREAST_CANCER_DATASET = './data/BCW/wdbc.csv'

  rawData = pd.read_csv(BREAST_CANCER_DATASET, header=None)
  answers = rawData.loc[:,1]
  translate = { "M": 1, "B": 0}
  answers = answers.replace(translate)

  BCWdata = rawData.drop([0,1],axis=1)

  #now scale it
  scaler = StandardScaler()

  train_data, test_data, train_labels, test_labels = train_test_split(BCWdata, answers, test_size=0.25)

  #according to the following, the scaling must be done afterwards:
  # https://www.kaggle.com/code/graymant/breast-cancer-diagnosis-with-pytorch

  train_data = scaler.fit_transform(train_data)
  test_data = scaler.fit_transform(test_data)

  bcw_train = TensorDataset(torch.from_numpy(train_data).float(),
                          torch.reshape(torch.from_numpy(train_labels.to_numpy()),(426,1)).float())
  bcw_test = TensorDataset(torch.from_numpy(test_data).float(), 
                          torch.reshape(torch.from_numpy(test_labels.to_numpy()),(143,1)).float())
  BCW_TEST_SET = DataLoader(bcw_test, batch_size=10, shuffle=True)
  
  datasets = []
  users = 0
  for images, labels in bcw_loader:
    client_dataset = ClientSet(images, labels)
    datasets.append(DataLoader(client_dataset, batch_size = 128, shuffle=True)) #small batch size because there are only 85 images per user with 5 users.  This makes it 7 batches.
    users += 1
    if(users >= num_users):
      return datasets, bcw_test
  return datasets, bcw_test

#### Shakespeare

#dataset was made using the preprocess.sh -> shakespeare.py process detailed in the paper the LSTM is from.  Only difference is that I saved it using torch instead.

#source: https://github.com/Accenture/Labs-Federated-Learning/tree/free-rider_attacks

#assemble the sets!
def makeSpeareData(num_users):
  SHAKESPEARE_TRAIN_DATASET = './data/SPEARE/Shakespeare_train.pt'
  SHAKESPEARE_TEST_DATASET = './data/SPEARE/Shakespeare_test.pt'
  if((not exists(SHAKESPEARE_TRAIN_DATASET)) or (not exists(SHAKESPEARE_TEST_DATASET))):
    shakespeare.makeSpeareData()
  
  Speare_train_sets = torch.load(SHAKESPEARE_TRAIN_DATASET)
  Speare_test_sets = torch.load(SHAKESPEARE_TRAIN_DATASET)

  speare_test = TensorDataset(torch.tensor(Speare_test_sets[0][0]), torch.squeeze(torch.tensor(Speare_test_sets[1][0])))
  SPEARE_TEST_SET = DataLoader(speare_test, batch_size=512, shuffle=False)

  datasets = []
  if num_users > 10:
    num_users = 10 #there just aren't more than 10 sets, so cap it.
  for user in range(num_users):
    #get the input and labels
    c_input = torch.tensor(Speare_train_sets[0][user])
    c_labels = torch.squeeze(torch.tensor(Speare_train_sets[1][user]))
    #print(c_input.shape, c_labels.shape)
    c_dataset = TensorDataset(c_input, c_labels)
    datasets.append(DataLoader(c_dataset, batch_size=512, shuffle=True))
    if(users >= num_users):
      return datasets, speare_test
  return datasets, speare_test

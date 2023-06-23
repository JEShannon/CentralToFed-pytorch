from datasetBase import ClientSet
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd

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

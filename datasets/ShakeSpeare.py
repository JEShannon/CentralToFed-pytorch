from datasetBase import ClientSet
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from os.path import exists
from databuilders.Speare import shakespeare

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

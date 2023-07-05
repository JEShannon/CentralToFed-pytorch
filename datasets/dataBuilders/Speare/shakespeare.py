#### NOTE: THIS FILE WAS TAKEN FROM THE FOLLOWING REPOSITORY WITH SOME MINOR ALTERATIONS!
#
#  https://github.com/Accenture/Labs-Federated-Learning/tree/free-rider_attacks
#
#### THIS FILE IS NOT MY WORK!

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
import string
import numpy as np
import torch
import os
import sys

n_clients = 5

# In[2]:
# OPEN THE FILES CREATED BY LEAF
myLoc = os.path.dirname(sys.modules[__name__].__file__)
file_train = "all_data_iid_01_2_keep_0_train_9.json"
file_test = "all_data_iid_01_2_keep_0_test_9.json"


# In[2]:
def conversion_string_to_vector(sentence):

    all_characters = string.printable

    vector = [all_characters.index(sentence[c]) for c in range(len(sentence))]

    return vector


# In[2]:
def create_clients(data, n_clients):
    clients_X = []
    clients_y = []

    for client in range(n_clients):

        client_X = []
        client_y = []

        dic_client = data["user_data"][data["users"][client]]
        print(data["num_samples"][client])
        X, y = dic_client["x"], dic_client["y"]

        for X_i, y_i in zip(X, y):

            client_X.append(conversion_string_to_vector(X_i))
            client_y.append(conversion_string_to_vector(y_i))

        clients_X.append(client_X)
        clients_y.append(client_y)

    return clients_X, clients_y

def makeSpeareData():
    with open(os.path.join(myLoc, f"train/{file_train}")) as json_file:
        data_train = json.load(json_file)
    with open(os.path.join(myLoc, f"test/{file_test}")) as json_file:
        data_test = json.load(json_file)

    train_path = os.path.join(myLoc, os.pardir, os.pardir, "data/SPEARE/Shakespeare_train.pt")
    print(train_path)
    torch.save(create_clients(data_train, 10), train_path)

    test_path = os.path.join(myLoc, os.pardir, os.pardir, "data/SPEARE/Shakespeare_test.pt")
    torch.save(create_clients(data_test, 1), test_path)


# In[2]:

#clients_X_train, clients_y_train = create_clients(data_train, 5)
#clients_X_test, clients_y_test = create_clients(data_test, 5)

#n_samples = [len(clients_X_train[k]) + len(clients_X_test[k]) for k in range(n_clients)]
#print(sum(n_samples), np.mean(n_samples), np.std(n_samples))

from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, MNIST
from torchvision.transforms import transforms
import torch
from torch.distributions.dirichlet import Dirichlet
import random
from collections import Counter
import numpy as np
from typing import List
import pandas as pd
import pickle
import os

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

NUM_PACKETS = 20
NUM_FEATURES = 128

def data_processing(df,NUM_PACKETS, NUM_FEATURES):
   y_train = df['Label']
   flow_id = df['flow_id']

   df = df/255

   X_train = df.drop(['Label', 'flow_id'], axis=1)
   X_train = X_train.to_numpy()

   X_train = X_train.reshape(-1,NUM_PACKETS, NUM_FEATURES)
   y_train = y_train.to_numpy()

   y_train = y_train.reshape(-1,NUM_PACKETS)[:,-1]
   return X_train, y_train

#define ten dataset vt thuong :V
def load_data(dataset: str):
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10("dataset/cifar10", train=True, download=True, transform=train_transform)
        testset = CIFAR10("dataset/cifar10", train=False, download=True, transform=test_transform)

    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transform.Normalize((0.5), (0.5))
        ])

        trainset = MNIST('dataset/mnist', split='balanced', train=True, download=True, transform=transform)
        testset = MNIST('dataset/mnist', split='balanced', train=False, download=True, transform=transform)

    elif dataset == 'etc':
        #gquic data with additinal data 
        file_path = os.path.join(os.getcwd(), "dataset", "ETC", "etc_tensor_fl_data.pkl")
        with open(file_path, "rb") as f:
            loaded_dataset = pickle.load(f)
        trainset = {k: v for k, v in loaded_dataset.items() if k != "test"}
        testset = loaded_dataset['test']
    elif dataset == 'etc256':
        file_path = os.path.join(os.getcwd(), "dataset", "ETC", "etc_tensor_fl_data_256.pkl")
        with open(file_path, "rb") as f:
            loaded_dataset = pickle.load(f)
        trainset = {k: v for k, v in loaded_dataset.items() if k != "test"}
        testset = loaded_dataset['test']
    # elif dataset == "emnist":
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5), (0.5))
    #     ])

    #     trainset = EMNIST("data", split="balanced", train=True, download=True, transform=transform)
    #     testset = EMNIST("data", split="balanced", train=False, download=True, transform=transform)
    
    # elif dataset == "fmnist":
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5), (0.5))
    #     ])

    #     trainset = FashionMNIST(root='data', train=True, download=True, transform=transform)
    #     testset = FashionMNIST(root='data', train=True, download=True, transform=transform)
    return trainset, testset


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist

def dirichlet_data(trainset, num_clients: int, num_iids: int, alpha: float, beta: float):
    """
    Chia dữ liệu thành Non-IID partitions theo phân phối Dirichlet

    Parameters:
    - trainset: Tập dữ liệu
    - num_clients: Số lượng clients.
    - num_iids: Số lượng clients iid 
    - alpha: Hệ số Dirichlet của client iid (100, 1000,....)
    - beta: Hệ số Dirichlet của client non-iid (0.1, 0.2,...)

    Returns:
    - ids: Danh sách indices của dữ liệu cho mỗi client.
    - label_dist: Phân phối nhãn của mỗi client.
    """
    classes = list(set(trainset.classes))
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]

    ids = [[] for _ in range(num_clients)]
    label_dist = []
    labels = list(range(len(classes)))

    for i in range(num_clients):
        concentration = torch.ones(len(labels))*alpha if i < num_iids else torch.ones(len(labels))*beta
        dist = Dirichlet(concentration).sample()
        for _ in range(client_size):
            label = random.choices(labels, dist)[0]
            id = random.choices(data[label])[0]
            ids[i].append(id)
            data[label].remove(id)

            if len(data[label]) == 0:
                dist = renormalize(dist, labels, label)
                labels.remove(label)

        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i) for i in range(len(classes))})

    return ids, label_dist


def shard_data(trainset, num_clients: int, num_shards: int):
    """
    Chia dữ liệu thành Non-IID partitions theo phương pháp Shards.

    Parameters:
    - trainset: Tập dữ liệu
    - num_clients: Số lượng clients.
    - num_shards: Số lượng shards để chia.

    Returns:
    - ids: Danh sách indices của dữ liệu cho mỗi client.
    - label_dist: Phân phối nhãn của mỗi client.
    """
    classes = list(set(trainset.classes))  # Danh sách các lớp (labels)
    num_classes = len(classes)
    
    # Gom nhóm dữ liệu theo nhãn
    data_by_label = {label: [] for label in range(num_classes)}
    for i in range(len(trainset)):
        data_by_label[trainset[i][1]].append(i)

    # Chia mỗi nhãn thành các shards nhỏ
    shards = []
    for label in data_by_label:
        np.random.shuffle(data_by_label[label])  # Xáo trộn dữ liệu của nhãn
        num_samples_per_shard = len(data_by_label[label]) // (num_shards // num_classes)
        shards.extend(np.split(np.array(data_by_label[label]), len(data_by_label[label]) // num_samples_per_shard))

    np.random.shuffle(shards)  # Xáo trộn danh sách shards

    # Gán shards cho mỗi client
    num_shards_per_client = num_shards // num_clients
    ids = [[] for _ in range(num_clients)]
    label_dist = []

    for i in range(num_clients):
        assigned_shards = shards[i * num_shards_per_client : (i + 1) * num_shards_per_client]
        for shard in assigned_shards:
            ids[i].extend(shard.tolist())

        # Thống kê phân bố nhãn của client
        counter = Counter([trainset[idx][1] for idx in ids[i]])
        label_dist.append({classes[j]: counter.get(j, 0) for j in range(num_classes)})

    return ids, label_dist

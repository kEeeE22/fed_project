import torch
from typing import List
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# def set_parameters(net, parameters: List[np.ndarray]):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=False)
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict()

    for k, v in params_dict:
        if isinstance(v, np.ndarray):  # Chuyển từ numpy sang tensor
            state_dict[k] = torch.tensor(v, dtype=torch.float32)
        elif isinstance(v, np.float32) or isinstance(v, float):  # Xử lý lỗi float32
            state_dict[k] = torch.tensor([v], dtype=torch.float32)  # Chuyển thành tensor 1 chiều
        else:
            state_dict[k] = v  # Nếu đã là tensor thì giữ nguyên

    net.load_state_dict(state_dict, strict=False)

def train(net, trainloader, epochs, lr,frozen=False, proximal_mu=None):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0002, verbose=False)
    global_params = copy.deepcopy(net).parameters()

    if frozen:
      for name, param in net.named_parameters():
        if "bic" in name:
            param.requires_grad = False
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    #training
    net.train()
    for epoch in range(epochs):
        total_loss, correct, total_samples = 0.0, 0, 0
        for images, labels in trainloader:
          #images, labels = batch["img"], batch["label"]
          images, labels = images.to(DEVICE), labels.to(DEVICE)

          optimizer.zero_grad()
          outputs = net(images)
          if proximal_mu != None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
          else:
            loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()

          # Tính loss
          total_loss += loss.item()

          # Tính accuracy
          _, preds = torch.max(outputs, 1)
          correct += (preds == labels).sum().item()
          total_samples += labels.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        early_stopping(epoch_loss)
        if early_stopping.early_stop:
          print("Dừng sớm do không cải thiện!")
          break

def trainbic(net, trainloader, epochs, lr,frozen=False, proximal_mu=None):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0002, verbose=False)
    global_params = copy.deepcopy(net).parameters()

    if frozen:
        for name, param in net.named_parameters():
            if "bic" not in name:
                param.requires_grad = False
        net.bic.alpha.requires_grad = True
        net.bic.alpha.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    #training
    net.train()
    for epoch in range(epochs):
        total_loss, correct, total_samples = 0.0, 0, 0
        for images, labels in trainloader:
          #images, labels = batch["img"], batch["label"]
          images, labels = images.to(DEVICE), labels.to(DEVICE)

          optimizer.zero_grad()
          outputs = net(images)
          if proximal_mu != None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
          else:
            loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()

          # Tính loss
          total_loss += loss.item()

          # Tính accuracy
          _, preds = torch.max(outputs, 1)
          correct += (preds == labels).sum().item()
          total_samples += labels.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = correct / total_samples
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        early_stopping(epoch_loss)
        if early_stopping.early_stop:
          print("Dừng sớm do không cải thiện!")
          break

def test(net,testloader):
    """Evaluate the network on the entire test set."""
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            #images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader)
    accuracy = correct / total
    return loss, accuracy


def test_2_server(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    all_labels = []
    all_preds = []

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            #images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())  
            all_preds.extend(predicted.cpu().numpy())  

    loss /= len(testloader)
    accuracy = correct / total

    # Tạo classification report
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=1)
    df = pd.DataFrame(report_dict).transpose()

    return loss, accuracy, report_dict['macro avg']['precision'], report_dict['macro avg']['recall'], report_dict['macro avg']['f1-score']
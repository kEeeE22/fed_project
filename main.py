import torch
import argparse
import random
import numpy as np
import os
import time
import pickle
from torch.utils.data import DataLoader, Subset, TensorDataset

from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

from sklearn.model_selection import train_test_split 

from utils.utils1 import train, get_parameters
from utils.load_data import load_data, dirichlet_data, shard_data
from utils.model import CNN1, ETC_CNN, ResNet50, ETC_CNN2, ETC_CNN3, ETC_RESNET18, ETC_CNN1D, ETC_CNN_0_1

from fedbic.bic_client import BiCClient
from fedbic.bic_strategy import FedBic
from fedbic.old_bic_client import OLD_BiCClient
from fedbic.wb_bic_client import WB_BiCClient

from baseline.avg_strategy import FedAvg, weighted_average
from baseline.client import BaselineClient

from baseline.BN_client import BNClient
from baseline.BN_strategy import FedBN

from baseline.prox_client import ProxClient
from baseline.prox_strategy import FedProx

def setup_logger():
    os.makedirs("results", exist_ok=True)
    return open("results/training_log.txt", "a")

log_file = setup_logger()

def log_time(stage, start_time):
    elapsed_time = time.time() - start_time
    log_message = f"[LOG] {stage} completed in {elapsed_time:.2f} seconds."
    print(log_message)
    log_file.write(log_message + "\n")
    log_file.flush()


def fed_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mt', '--method', type=str, required=True, help='Method name')
    parser.add_argument('-rd', '--num-round', type=int, required=True, help='Number of rounds')
    parser.add_argument('-ds', '--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('-md', '--sys-model', type=str, required=True, help='Model name')
    parser.add_argument('-is', '--sys-i-seed', type=int, required=True, default=42, help='Seed used in experiment')
    
    #client
    parser.add_argument('-nc', '--n-client', type=int, required=True, help='Number of clients')
    parser.add_argument('-eps', '--client-epochs', type=int, required=True,default=1, help='Client epochs for federated learning')
    parser.add_argument('-lr', '--client-lr', required=True, type=float, help='Client learning rate')

    #phase 1 // chi can neu method la FedBic
    parser.add_argument('-p1-eps', '--phase-1-client-epochs', type=int, help='BiC training epochs')
    parser.add_argument('-p1-lr', '--phase-1-client-lr',type=float, help='BiC training learning rate')
    parser.add_argument('-mode', '--bic_mode', type=str, help='Mode for BiCClient')
    #mu cang cao cang khac fedavg
    parser.add_argument('-mu', '--proximal-mu', type=float, help='proximal mu for fedprox')

    #dirichlet
    parser.add_argument('-niid', '--num-iid',type=int, help='Number of iid clients')
    parser.add_argument('-alpha', '--alpha',type=float, help='Dirichlet parameter for iids')
    parser.add_argument('-beta', '--beta',type=float, help='Dirichlet parameter for non-iids')

    #shard
    parser.add_argument('-ns', '--num-shard', type=int, help='Number of shard')
    args = parser.parse_args()

    if args.method.lower() == "oldfedbic":
        if args.phase_1_client_epochs is None or args.phase_1_client_lr is None:
            parser.error("Arguments -p1-eps and -p1-lr are required when method is 'fedbic'")
    if args.method.lower() == 'fedprox':
        if args.proximal_mu is None:
            parser.error("Arguments -mu is required when method is 'fedprox' ")

    #kiem tra xem dung dirichlet hay shard de chia du lieu
    using_etc = args.dataset == 'etc'
    using_dirichlet = args.alpha is not None and args.beta is not None and args.num_iid is not None
    using_shard = args.num_shard is not None
    if not using_etc and not (using_dirichlet or using_shard):
        parser.error("You must specify either Dirichlet parameters (-alpha, -beta, -niid) or -shard for data partitioning, but not both.")

    # Không được chọn cả hai nếu dataset không phải 'etc'
    if not using_etc and using_dirichlet and using_shard:
        parser.error("You cannot specify both Dirichlet (-alpha, -beta, -niid) and Shard (-shard) methods. Choose one.")
    return args

def main():
    total_start_time = time.time()
    args = fed_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message = f"Training on {DEVICE}"
    print(log_message)
    log_file.write(log_message + "\n")
    
    start_time = time.time()
    train_set, test_set = load_data(args.dataset)
    log_time("Data loading", start_time)

    fraction_fit=1.0
    fraction_evaluate=1.0
    min_fit_clients=5
    min_evaluate_clients=5
    min_available_clients=5

    #dataset viet thuong het  :V
    dataset_list = ['mnist', 'cifar10', 'etc', 'etc256']
    assert args.dataset in dataset_list, 'Choose a dataset that exist.'



    #MODEL
    model_dict = {'CNN1': CNN1, 
                  'ETC_CNN': ETC_CNN, 
                  'RESNET50': ResNet50, 
                  'ETC_CNN2': ETC_CNN2, 
                  'ETC_CNN3': ETC_CNN3, 
                  'ETC_RESNET18': ETC_RESNET18, 
                  'ETC_CNN1D': ETC_CNN1D,
                  'ETC_CNN_0_1': ETC_CNN_0_1}
    assert args.sys_model in model_dict, 'Choose a model that exist'



    random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)


    os.makedirs("results", exist_ok=True)
    # file
    avg_file = f"results/avg_{args.method}_{args.num_round}_{args.sys_model}_{args.dataset}_{args.n_client}_{args.num_round}_{args.client_lr}_{args.beta}.csv"
    client_file = f'results/client_{args.method}_{args.num_round}_{args.sys_model}_{args.dataset}_{args.n_client}_{args.num_round}_{args.client_lr}_{args.beta}.csv'
    server_file = f'results/server_{args.method}_{args.num_round}_{args.sys_model}_{args.dataset}_{args.n_client}_{args.num_round}_{args.client_lr}_{args.beta}.csv'
    global_model_file = f'results/global_model_{args.method}_{args.num_round}_{args.sys_model}_{args.dataset}_{args.n_client}_{args.num_round}_{args.client_lr}_{args.beta}.pth'



    start_time = time.time()
    if args.dataset == "etc":
        ids, labels = None, None  # Không chia dữ liệu nếu dataset là "etc"
    elif args.alpha is not None and args.beta is not None and args.num_iid is not None:
        ids, labels = dirichlet_data(train_set, args.n_client, args.num_iid, args.alpha, args.beta)
    elif args.num_shard is not None:
        ids, labels = shard_data(train_set, args.n_client, args.num_shard)
    log_time("Data partitioning", start_time)


    start_time = time.time()
    trainloaders = []
    valloaders = []
    if args.dataset == 'etc':
        for i in range(1, 7):
            dataa = train_set[f'client{i}']
            x, y = zip(*dataa)  
            
            x = torch.stack(x).float()  
            y = torch.tensor(y).long() 

            # Chia train/test
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

            x_train = x_train.unsqueeze(1)  
            x_val = x_val.unsqueeze(1)

            traina = TensorDataset(x_train, y_train)
            vala = TensorDataset(x_val, y_val)

            trainloaders.append(DataLoader(traina, batch_size=64, shuffle=True))
            valloaders.append(DataLoader(vala, batch_size=64, shuffle=False))
        x_test, y_test = zip(*test_set)  # Tách features và labels

        # Chuyển danh sách thành tensor
        x_test = torch.stack(x_test).float().unsqueeze(1)  # Thêm chiều kênh: (batch_size, 1, height, width)
        y_test = torch.tensor(y_test).long()  # Đảm bảo labels là LongTensor

        # Gán lại vào dataset
        server_set = TensorDataset(x_test, y_test)

        # Sau đó tạo DataLoader bình thường
        server_test = DataLoader(server_set, batch_size=64, shuffle=True)
    else:
        for i in range(args.n_client):
            num_samples = len(ids[i])
            num_val = int(0.2 * num_samples)
            num_train = num_samples - num_val
            train_indices = ids[i][:num_train]
            val_indices = ids[i][num_train:]
            trainloaders.append(DataLoader(Subset(train_set, train_indices), batch_size=64, shuffle=True))
            valloaders.append(DataLoader(Subset(train_set, val_indices), batch_size=64, shuffle=False))
        server_test = DataLoader(test_set, batch_size=64, shuffle=False)
    log_time("DataLoader initialization", start_time)



    start_time = time.time()
    #choose method
    if args.method == 'FedBic':
        if args.mode == 'wb' or args.mode == 'nonwb':
            bic_params = []
            for i in range(args.n_client):
                net = model_dict[args.sys_model]().to(DEVICE)
                train(net, trainloaders[i], epochs=args.phase_1_client_epochs, lr=args.phase_1_client_lr)

                bic_ = get_parameters(net)
                bic_arrays = bic_[-1]
                #bic_arrays = parameters_to_ndarrays(bic_)
                bic_params.append(bic_arrays)
                print('Done append bic params client ' + str(i))

        def client_fn(context: Context) -> Client:
            net = model_dict[args.sys_model]().to(DEVICE)
            partition_id = context.node_config["partition-id"]
            num_partitions = args.n_client
            trainloader = trainloaders[partition_id]
            valloader = valloaders[partition_id]
            epochs = args.client_epochs
            client_lr = args.client_lr
            num_rounds = args.num_round
            mode = args.bic_mode
            if args.mode == 'wb' or args.mode == 'nonwb':
                bic_params_client = bic_params[partition_id]
                return WB_BiCClient(partition_id, net, trainloader, valloader, epochs, client_lr, bic_params_client, mode).to_client()
            elif args.mode == 'er' or args.mode == 'lr':
                return BiCClient(partition_id, net, trainloader, valloader, epochs,client_lr, num_rounds, mode).to_client()

        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=args.num_round)
            strategy = FedBic(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
                server_file = server_file,
                client_file = client_file,
                avg_file = avg_file
            )
            return ServerAppComponents(config=config, strategy = strategy)


        # Create ServerApp
        server = ServerApp(server_fn=server_fn)
    elif args.method == 'oldFedBic':

        def client_fn(context: Context) -> Client:
            net = model_dict[args.sys_model]().to(DEVICE)
            partition_id = context.node_config['partition-id']
            num_partitions = args.n_client
            trainloader = trainloaders[partition_id]
            valloader = valloaders[partition_id]
            epochs = args.client_epochs
            client_lr = args.client_lr
            return OLD_BiCClient(partition_id, net, trainloader, valloader, epochs, client_lr).to_client()

        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=args.num_round)
            strategy = FedBic(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
                server_file = server_file,
                client_file = client_file,
                avg_file = avg_file
            )
            return ServerAppComponents(config=config, strategy = strategy)


        # Create ServerApp
        server = ServerApp(server_fn=server_fn)
    elif args.method == 'FedAvg':
        def client_fn(context: Context) -> Client:
            net = model_dict[args.sys_model]().to(DEVICE)
            partition_id = context.node_config["partition-id"]
            num_partitions = args.n_client
            trainloader = trainloaders[partition_id]
            valloader = valloaders[partition_id]
            epochs = args.client_epochs
            client_lr = args.client_lr
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            return BaselineClient(partition_id, net, trainloader, valloader, epochs, client_lr).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=args.num_round)
            strategy = FedAvg(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
                server_file = server_file,
                client_file = client_file,
                avg_file = avg_file
            )
            return ServerAppComponents(config=config, strategy = strategy)


        # Create ServerApp
        server = ServerApp(server_fn=server_fn)
    elif args.method == 'FedBN':
        def client_fn(context: Context) -> Client:
            net = model_dict[args.sys_model]().to(DEVICE)
            partition_id = context.node_config["partition-id"]
            num_partitions = args.n_client
            trainloader = trainloaders[partition_id]
            valloader = valloaders[partition_id]
            epochs = args.client_epochs
            client_lr = args.client_lr
            bn_state_dir = "results/bn_state"
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            return BNClient(partition_id, net, trainloader, valloader, epochs, client_lr, bn_state_dir).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=args.num_round)
            strategy = FedBN(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
                server_file = server_file,
                client_file = client_file,
                avg_file = avg_file
            )
            return ServerAppComponents(config=config, strategy = strategy)


        # Create ServerApp
        server = ServerApp(server_fn=server_fn)
    elif args.method == 'FedProx':
        def client_fn(context: Context) -> Client:
            net = model_dict[args.sys_model]().to(DEVICE)
            partition_id = context.node_config["partition-id"]
            num_partitions = args.n_client
            trainloader = trainloaders[partition_id]
            valloader = valloaders[partition_id]
            epochs = args.client_epochs
            client_lr = args.client_lr
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            return ProxClient(partition_id, net, trainloader, valloader, epochs, client_lr).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=args.num_round)
            strategy = FedProx(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
                server_file = server_file,
                client_file = client_file,
                avg_file = avg_file,
                proximal_mu = args.proximal_mu
            )
            return ServerAppComponents(config=config, strategy = strategy)


        # Create ServerApp
        server = ServerApp(server_fn=server_fn)
    log_time("Server & strategy initialization", start_time)
    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_gpus": 1, "num_cpus": 1}}

    # Run simulation
    start_time = time.time()
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.n_client,
        backend_config=backend_config,
    )
    log_time("Federated Learning simulation", start_time)
    
    log_time("Total execution", total_start_time)
    log_file.close()

if __name__ == '__main__':
    main()


#python main.py -mt FedBic -rd 1000 -ds etc -md ETC_CNN -is 42 -nc 6 -eps 1 -lr 0.0001 -p1-eps 50 -p1-lr 0.001
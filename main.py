import torch
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader, Subset

from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

from utils.utils1 import train, get_parameters
from utils.load_data import load_data, dirichlet_data, shard_data
from utils.model import CNN1, ETC_CNN

from fedbic.bic_client import BiCClient
from fedbic.bic_strategy import FedBic

from baseline.avg_strategy import FedAvg, weighted_average
from baseline.client import BaselineClient

from baseline.BN_client import BNClient
from baseline.BN_strategy import FedBN

from baseline.prox_client import ProxClient
from baseline.prox_strategy import FedProx




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
    parser.add_argument('-lr', '--client-lr', required=True, help='Client learning rate')

    #phase 1
    parser.add_argument('-p1-eps', '--phase-1-client-epochs', type=int, required=True, help='BiC training epochs')
    parser.add_argument('-p1-lr', '--phase-1-client-lr',type=float, required=True, help='BiC training learning rate')

    #dirichlet
    parser.add_argument('-niid', '--num-iid',type=int, required=True, help='Number of iid clients')
    parser.add_argument('-alpha', '--alpha',type=float, required=True, help='Dirichlet parameter for iids')
    parser.add_argument('-beta', '--beta',type=float, required=True, help='Dirichlet parameter for non-iids')

    args = parser.parse_args()
    return args

def main():
    args = fed_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    dataset_list = ['MNIST', 'CIFAR10', 'ETC']
    assert args.dataset in dataset_list, 'Choose a dataset that exist.'

    model_dict = {'CNN1': CNN1, 'ETC_CNN': ETC_CNN}
    assert args.sys_model in model_dict, 'Choose a model that exist'

    random.seed(args.sys_i_seed)
    torch.manual_seed(args.sys_i_seed)
    np.random.seed(args.sys_i_seed)

    train_set, test_set = load_data(args.dataset)
    ids, labels = dirichlet_data(train_set, args.n_client, args.num_iid, args.alpha, args.beta)

    # file
    avg_file = f"avg_{args.method}_{args.num_round}_{args.client_lr}_{args.beta}.csv"
    client_file = f'client_{args.method}_{args.num_round}_{args.client_lr}_{args.beta}.csv'
    server_file = f'server_{args.method}_{args.num_round}_{args.client_lr}_{args.beta}.csv'

    trainloaders = []
    valloaders = []

    for i in range(args.n_client):
        num_samples = len(ids[i])
        num_val = int(0.2 * num_samples)
        num_train = num_samples - num_val

        train_indices = ids[i][:num_train]
        val_indices = ids[i][num_train:]

        trainloaders.append(DataLoader(Subset(train_set, train_indices), batch_size=64, shuffle=True))
        valloaders.append(DataLoader(Subset(train_set, val_indices), batch_size=64, shuffle=False))

    server_test = DataLoader(test_set, batch_size=64, shuffle=False)

    #choose method
    if args.method == 'FedBic':
        bic_params = []
        for i in range(args.n_client):
            net = model_dict[args.sys_model]().to(DEVICE)
            train(net, trainloaders[i], epochs=args.phase_1_client_epochs, lr=args.phase_1_client_lr)

            bic_ = get_parameters(net)
            bic_arrays = bic_[-1]
            #bic_arrays = parameters_to_ndarrays(bic_)
            bic_params.append(bic_arrays)
            print('done append bic params client ' + str(i))

        def client_fn(context: Context) -> Client:
            net = model_dict[args.sys_model]().to(DEVICE)
            partition_id = context.node_config["partition-id"]
            num_partitions = args.n_client
            trainloader = trainloaders[partition_id]
            valloader = valloaders[partition_id]
            epochs = args.client_epochs
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            bic_params_client = bic_params[partition_id]
            return BiCClient(partition_id, net, trainloader, valloader, epochs, bic_params_client).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=20)
            strategy = FedBic(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=10,
                min_evaluate_clients=10,
                min_available_clients=10,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
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
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            bic_params_client = bic_params[partition_id]
            return BaselineClient(partition_id, net, trainloader, valloader, epochs, bic_params_client).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=20)
            strategy = FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=10,
                min_evaluate_clients=10,
                min_available_clients=10,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
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
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            bic_params_client = bic_params[partition_id]
            return BNClient(partition_id, net, trainloader, valloader, epochs, bic_params_client).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=20)
            strategy = FedBN(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=10,
                min_evaluate_clients=10,
                min_available_clients=10,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
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
            #epochs = random.randint(1,5)
            #epochs = client_epochs.get(f"client_{partition_id}", 1)
            bic_params_client = bic_params[partition_id]
            return ProxClient(partition_id, net, trainloader, valloader, epochs, bic_params_client).to_client()


        # Create the ClientApp
        client = ClientApp(client_fn=client_fn)

        def server_fn(context: Context) -> ServerAppComponents:
        # Configure the server for num_rounds rounds of training
            config = ServerConfig(num_rounds=20)
            strategy = FedProx(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=10,
                min_evaluate_clients=10,
                min_available_clients=10,
                evaluate_metrics_aggregation_fn=weighted_average,
                testloader = server_test,
                net = model_dict[args.sys_model](),
            )
            return ServerAppComponents(config=config, strategy = strategy)


        # Create ServerApp
        server = ServerApp(server_fn=server_fn)
    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_gpus": 1, "num_cpus": 1}}

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=args.n_client,
        backend_config=backend_config,
    )

if __name__ == '__main__':
    main()
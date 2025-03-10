from flwr.server.strategy import Strategy
from typing import Callable, List, Tuple, Optional, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggragate import aggregate, weighted_loss_avg
import os
import csv
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Metrics,
)
from utils.utils1 import set_parameters, get_parameters, test
from main import client_file, server_file, avg_file, DEVICE


class FedAvg(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 10,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 10,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Metrics]]], Metrics]
        ] = None,
        testloader = None,
        net = None
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.testloader = testloader
        self.net = net
    def __repr__(self) -> str:
        return "FedAvg"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        net = self.net
        ndarrays = get_parameters(net)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
      sample_size, min_num_clients = self.num_fit_clients(
        client_manager.num_available()
    )
      clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )

      standard_config = {"server_round": server_round}

      return [
        (client, FitIns(parameters, standard_config))
        for client in clients
      ]



    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        array_param = aggregate(weights_results)

        #final_params = get_parameters(net)
        final_params = ndarrays_to_parameters(array_param)
        return final_params, {}


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        return [(client, EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        client_accuracies = {evaluate_res.metrics["cid"]: evaluate_res.metrics["accuracy"] for _, evaluate_res in results}
        print(client_accuracies)
        # In ra kết quả accuracy của từng client
        print(f"Round {server_round} - Client Accuracies:")
        for client_id, acc in client_accuracies.items():
          print(f"  Client {client_id}: {acc:.2f}%")

        #client_file = f"client_{self.__repr__()}_{client_epochs}_{client_lr}_{alpha}.csv"
        file_exists = os.path.isfile(client_file)

        with open(client_file, mode="a", newline="") as f:
          writer = csv.writer(f)

          if not file_exists:
              writer.writerow(["Round", "Client ID", "Accuracy"])

          for client_id, acc in client_accuracies.items():
              writer.writerow([server_round, client_id, acc])


        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(evaluate_res.num_examples, evaluate_res.metrics) for _, evaluate_res in results]
            )

            # Tên file CSV
            #avg_file = f"avg_{self.__repr__()}_{client_epochs}_{client_lr}_{alpha}.csv"
            file_exists = os.path.isfile(avg_file)

            # Lưu kết quả vào file CSV
            with open(avg_file, mode="a", newline="") as f:
                writer = csv.writer(f)

                # Ghi header nếu file chưa tồn tại
                if not file_exists:
                    writer.writerow(["Method","Round", "Accuracy"])

                # Lấy tên phương pháp (hàm được gọi)
                method_name = self.__repr__()

                # Lưu kết quả vào file
                writer.writerow([method_name,server_round,metrics_aggregated['accuracy']])
        else:
            metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        array_param = parameters_to_ndarrays(parameters)
        net = self.net.to(DEVICE)
        set_parameters(net, array_param)
        loss, accuracy = test(net, self.testloader)

        # Tên file CSV
        #server_file = f"server_{self.__repr__()}_{client_epochs}_{client_lr}_{alpha}.csv"
        file_exists = os.path.isfile(server_file)

        # Lưu kết quả vào file CSV
        with open(server_file, mode="a", newline="") as f:
          writer = csv.writer(f)

          # Ghi header nếu file chưa tồn tại
          if not file_exists:
            writer.writerow(["Method","Round", "Accuracy"])

          # Lấy tên phương pháp (hàm được gọi)
          method_name = self.__repr__()

          # Lưu kết quả vào file
          writer.writerow([method_name,server_round, accuracy])
        return loss, {"accuracy": accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
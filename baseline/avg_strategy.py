import torch
from flwr.server.strategy import Strategy
from typing import Callable, List, Tuple, Optional, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
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
from utils.utils1 import set_parameters, get_parameters, test, test_2_server
#from main import client_file, server_file, avg_file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        net = None,
        server_file = None,
        client_file = None,
        avg_file = None,
        model_file = None,
        num_rounds = 100,
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
        self.server_file = server_file
        self.client_file = client_file
        self.avg_file = avg_file
        self.model_file = model_file
        self.num_rounds = num_rounds    
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
        if server_round == self.num_rounds:
            torch.save(final_params, self.model_file)
            print(f"Saved final global model at round {server_round} to pth")
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
        standard_config = {"server_round": server_round}
        return [(client, EvaluateIns(parameters, standard_config)) for client in clients]

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
        file_exists = os.path.isfile(self.client_file)

        with open(self.client_file, mode="a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["Round", "ID","Loss", "Accuracy", "Precision", "Recall", "F1-Score"])
        
            for _, evaluate_res in results:
                writer.writerow([
                    server_round,
                    evaluate_res.metrics["cid"],
                    evaluate_res.loss,
                    evaluate_res.metrics["accuracy"],
                    evaluate_res.metrics["precision"],
                    evaluate_res.metrics["recall"],
                    evaluate_res.metrics["f1_score"]
                ])


        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                [(evaluate_res.num_examples, evaluate_res.metrics) for _, evaluate_res in results]
            )
            print("Aggregated Metrics:", metrics_aggregated)
            # Tên file CSV
            #avg_file = f"avg_{self.__repr__()}_{client_epochs}_{client_lr}_{alpha}.csv"
            file_exists = os.path.isfile(self.avg_file)

            # Lưu kết quả vào file CSV
            with open(self.avg_file, mode="a", newline="") as f:
                writer = csv.writer(f)

                # Ghi header nếu file chưa tồn tại
                if not file_exists:
                    writer.writerow(["Round", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"])

                # Lấy tên phương pháp (hàm được gọi)
                #method_name = self.__repr__()
                writer.writerow([
                    server_round,
                    metrics_aggregated.get("loss", None),       # Tránh lỗi KeyError
                    metrics_aggregated.get("accuracy", None),
                    metrics_aggregated.get("precision", None),
                    metrics_aggregated.get("recall", None),
                    metrics_aggregated.get("f1_score", None),
                ])
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
        loss, accuracy, precision, recall, f1_score = test_2_server(net, self.testloader)

        # Tên file CSV
        #server_file = f"server_{self.__repr__()}_{client_epochs}_{client_lr}_{alpha}.csv"
        file_exists = os.path.isfile(self.server_file)
        
        # Lưu kết quả vào file CSV
        with open(self.server_file, mode="a", newline="") as f:
          writer = csv.writer(f)

        # Mở file CSV và ghi dữ liệu
        with open(self.server_file, mode="a", newline="") as f:
            writer = csv.writer(f)

            # Ghi header nếu file chưa tồn tại
            if not file_exists:
                writer.writerow(["Round", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"])

            # Ghi kết quả của lần đánh giá này
            writer.writerow([server_round, loss, accuracy, precision, recall, f1_score])
        return loss, {"accuracy": accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    


# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)

    # Nếu không có mẫu nào, tránh lỗi chia cho 0
    if total_examples == 0:
        return {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}

    # Tính trung bình có trọng số cho từng metric
    weighted_metrics = {
        "loss": sum(num_examples * m["loss"] for num_examples, m in metrics) / total_examples,
        "accuracy": sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples,
        "precision": sum(num_examples * m["precision"] for num_examples, m in metrics) / total_examples,
        "recall": sum(num_examples * m["recall"] for num_examples, m in metrics) / total_examples,
        "f1_score": sum(num_examples * m["f1_score"] for num_examples, m in metrics) / total_examples,
    }

    return weighted_metrics
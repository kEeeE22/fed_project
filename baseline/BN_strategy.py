from typing import Callable, List, Tuple, Optional, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
import os
import csv
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from baseline.avg_strategy import FedAvg


class FedBN(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return "FedBN"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Chỉ gửi các tham số không thuộc BatchNorm layers
        parameters_filtered = self.filter_out_batchnorm_layers(parameters)

        standard_config = {"server_round": server_round}

        return [(client, FitIns(parameters_filtered, standard_config)) for client in clients]

    def filter_out_batchnorm_layers(self, parameters: Parameters) -> Parameters:
        """Loại bỏ BatchNorm layers khỏi tham số của mô hình."""
        ndarrays = parameters_to_ndarrays(parameters)
        filtered_params = [
            param for name, param in zip(self.model_keys, ndarrays)
            if "bn" not in name.lower()  # Loại bỏ BatchNorm layers
        ]
        return ndarrays_to_parameters(filtered_params)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Chỉ tổng hợp các tham số không phải BatchNorm
        weights_results = [
            (self.filter_out_batchnorm_layers(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        array_param = aggregate(weights_results)

        # Chuyển đổi lại thành Parameters object
        final_params = ndarrays_to_parameters(array_param)
        return final_params, {}
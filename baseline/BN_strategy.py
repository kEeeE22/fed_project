from typing import Callable, List, Tuple, Optional, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from baseline.avg_strategy import FedAvg
import numpy as np

def get_parameters_BN(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for name, val in net.state_dict().items() if "bn" not in name]
class FedBN(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 10,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 10,
        **kwargs
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )

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

        aggregated_weights = aggregate(weights_results)
        
        # Chỉ tổng hợp trọng số của các lớp fully connected, giữ nguyên BatchNorm
        # previous_weights = get_parameters_BN(self.net)
        # for i, (prev, new) in enumerate(zip(previous_weights, aggregated_weights)):
        #     if "bn" in self.net.state_dict().keys()[i]:  # Bỏ qua BatchNorm
        #         aggregated_weights[i] = prev

        final_params = ndarrays_to_parameters(aggregated_weights)
        return final_params, {}
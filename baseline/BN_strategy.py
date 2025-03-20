# from typing import Callable, List, Tuple, Optional, Union, Dict
# from flwr.server.client_manager import ClientManager
# from flwr.server.client_proxy import ClientProxy
# from flwr.server.strategy.aggregate import aggregate

# from flwr.common import (
#     FitIns,
#     FitRes,
#     Parameters,
#     Scalar,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )
from baseline.avg_strategy import FedAvg


class FedBN(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __repr__(self) -> str:
        return 'FedBN'
from typing import Callable, List, Tuple, Optional, Union, Dict
import os
import csv
import torch
import torch.nn as nn  # Thêm import nn

from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    Parameters,
    Scalar,
)
from baseline.avg_strategy import FedAvg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedBN(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return 'FedBN'

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

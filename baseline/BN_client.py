from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import NDArrays
from typing import Dict
import torch
from collections import OrderedDict
from utils.utils1 import get_parameters, train
from baseline.client import BaselineClient
import os
import pickle

class BNClient(BaselineClient):
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr, bn_state_dir):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
        bn_state_dir = os.path.join(bn_state_dir, f"client_{partition_id}")
        os.makedirs(bn_state_dir, exist_ok=True)
        self.bn_state_pkl = os.path.join(bn_state_dir, f"client_{partition_id}.pkl")

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if.

        available.
        """
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

        # Now also load from bn_state_dir
        if os.path.exists(self.bn_state_pkl):  # It won't exist in the first round
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict()
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def _save_bn_statedict(self) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.tensor]:
        """Load pickle with BN state_dict and return as dict."""
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_state_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_state_dict
    
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] FedBN fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        self.set_parameters(ndarrays_original)
        
        train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)
        ndarrays_updated = get_parameters(self.net)

        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )
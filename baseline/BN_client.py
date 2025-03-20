from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateIns,
    EvaluateRes,
)
from flwr.common.typing import NDArrays
from typing import Dict, List
import torch
from collections import OrderedDict
from utils.utils1 import train, set_parameters, test_2_server
from baseline.client import BaselineClient
import os
import pickle
import numpy as np 



class BNClient(BaselineClient):
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr, bn_state_dir):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
        bn_state_dir = os.path.join(bn_state_dir, f"client_{partition_id}")
        os.makedirs(bn_state_dir, exist_ok=True)
        self.bn_state_pkl = os.path.join(bn_state_dir, f"client_{partition_id}.pkl")

    def set_parameters_BN(self, net, parameters: List[np.ndarray]) -> None:
      keys = [k for k in net.state_dict().keys() if "bn" not in k]
      
      # Load BatchNorm từ file `.pkl`
      bn_state_dict = self._load_bn_statedict()

      # Tạo OrderedDict từ các parameters được nhận
      state_dict = OrderedDict()
      for k, v in zip(keys, parameters):
          state_dict[k] = torch.tensor(v, dtype=torch.float32)
      
      # Gộp thêm BatchNorm nếu có
      state_dict.update(bn_state_dict)

      # Load vào model
      net.load_state_dict(state_dict, strict=False)

    def get_parameters(self, net) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict(net)
        # Excluding parameters of BN layers when using FedBN
        params = [
            val.cpu().numpy()
            for name, val in net.state_dict().items()
            if "bn" not in name
        ]
        return params

    def _save_bn_statedict(self,net) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in net.state_dict().items()
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
        if ins.config.get("server_round", 0) == 1:
            parameters_original = ins.parameters
            ndarrays_original = parameters_to_ndarrays(parameters_original)
            set_parameters(self.net, ndarrays_original)
        else:
            parameters_original = ins.parameters
            ndarrays_original = parameters_to_ndarrays(parameters_original)
            self.set_parameters_BN(self.net, ndarrays_original)
        
        train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)
        ndarrays_updated = self.get_parameters(self.net)

        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.partition_id}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # for i, param in enumerate(ndarrays_original):
        #   print(f"Param {i}: shape={param.shape}, dtype={param.dtype}")
          
        self.set_parameters_BN(self.net, ndarrays_original)
        loss, accuracy, precision, recall, f1_score = test_2_server(self.net, self.valloader)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy), "cid":self.partition_id, "precision": precision, "recall": recall, "f1_score": f1_score, "loss": loss},
        )
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import torch
from utils.utils1 import get_parameters, train
from baseline.client import BaselineClient


class BNClient(BaselineClient):
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] FedBN fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        
        for (name, param), new_param in zip(self.net.named_parameters(), ndarrays_original):
            if "bn" not in name:
                param.data.copy_(torch.tensor(new_param, dtype=param.data.dtype, device=param.data.device))
        # Huấn luyện mô hình
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
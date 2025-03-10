from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np
from typing import List
from flwr.client import Client

from utils.utils1 import get_parameters, set_parameters, train, test

class BaselineClient(Client):
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr):
        self.partition_id = partition_id #id client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs
        self.client_lr = client_lr

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.partition_id}] get_parameters")


        ndarrays: List[np.ndarray] = get_parameters(self.net)

        parameters = ndarrays_to_parameters(ndarrays)

        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)


        train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)
        ndarrays_updated = get_parameters(self.net)

        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics = {},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.partition_id}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_parameters(self.net, ndarrays_original)
        loss, accuracy = test(self.net, self.valloader)


        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy), "cid":self.partition_id},
        )
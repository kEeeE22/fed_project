from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from utils.utils1 import get_parameters, set_parameters, train
from baseline.client import BaselineClient


class ProxClient(BaselineClient):
  def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr):
    super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
    self.proximal_mu = None

  def fit(self, ins: FitIns) -> FitRes:
    print(f"[Client {self.partition_id}] fit, config: {ins.config}")

    parameters_original = ins.parameters
    ndarrays_original = parameters_to_ndarrays(parameters_original)
    set_parameters(self.net, ndarrays_original)

    self.proximal_mu = ins.config.get("proximal_mu")
    train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True, proximal_mu=self.proximal_mu)
    ndarrays_updated = get_parameters(self.net)

    parameters_updated = ndarrays_to_parameters(ndarrays_updated)

    status = Status(code=Code.OK, message="Success")
    return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics = {},
    )
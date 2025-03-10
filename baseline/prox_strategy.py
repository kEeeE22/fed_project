from baseline.avg_strategy import FedAvg
from typing import List, Tuple
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.common import (
    FitIns,
    Parameters,
)

class FedProx(FedAvg):
  def __init__(self, proximal_mu, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.proximal_mu = proximal_mu

  def __repr__(self) -> str:
    return 'FedProx'

  def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
      sample_size, min_num_clients = self.num_fit_clients(
        client_manager.num_available()
    )
      clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )

      standard_config = {
        "server_round": server_round,
        "proximal_mu": self.proximal_mu  # Truyền giá trị proximal_mu từ server xuống client
        }

      return [
        (client, FitIns(parameters, standard_config))
        for client in clients
      ]
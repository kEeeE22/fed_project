from baseline.avg_strategy import FedAvg


class FedBic(FedAvg):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def __repr__(self) -> str:
    return 'FedBic'
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
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] fit, config: {ins.config}")

        # Nhận tham số từ server nhưng không ghi đè BatchNorm layers
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Chỉ cập nhật trọng số không phải BatchNorm
        self.set_parameters_except_batchnorm(self.net, ndarrays_original)

        # Huấn luyện client bình thường
        train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)

        # Lấy lại tham số sau khi train xong
        ndarrays_updated = get_parameters(self.net)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )

    def set_parameters_except_batchnorm(self, model, parameters):
        """Cập nhật tham số từ server nhưng giữ nguyên BatchNorm layers."""
        state_dict = model.state_dict()
        param_idx = 0

        for name, param in state_dict.items():
            if "bn" not in name.lower():  # Chỉ cập nhật nếu không phải BatchNorm
                state_dict[name] = torch.tensor(parameters[param_idx])
                param_idx += 1

        model.load_state_dict(state_dict, strict=False)
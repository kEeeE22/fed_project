from baseline.client import BaselineClient
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from utils.utils1 import get_parameters, set_parameters, train, test_2_server

'''mode = x'''
class OLD_BiCClient(BaselineClient):
    #Ver này sẽ train model như bthg và giữ lại lớp bic để dùng cho eval và các round tiếp theo
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
        self.bic_params = None

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        if self.bic_params is not None:
            print("[INFO] Add bic.")
            ndarrays_original.append(self.bic_params)

        set_parameters(self.net, ndarrays_original)

        train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=False)

        #lay tham so
        modelr_ndarrays = get_parameters(self.net)
        # self.bic_prams = modelr_ndarrays[-1]
        # torch.save(self.bic_prams, bic_path)
        # print(f"[Client {self.partition_id}] Saved BiC Layer to {bic_path}")
        model_ndarrays = modelr_ndarrays[:-1]
        self.bic_params = modelr_ndarrays[-1]
        parameters_updated = ndarrays_to_parameters(model_ndarrays)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader.dataset),
            metrics = {},
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.partition_id}] evaluate, config: {ins.config}")
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        
        if self.bic_params is not None:
            ndarrays_original.append(self.bic_params)
        set_parameters(self.net, ndarrays_original)
        loss, accuracy, precision, recall, f1_score = test_2_server(self.net, self.valloader)


        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader.dataset),
            metrics={"accuracy": float(accuracy), "cid":self.partition_id, "precision": precision, "recall": recall, "f1_score": f1_score, "loss": loss},
        )

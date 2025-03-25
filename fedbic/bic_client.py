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

from utils.utils1 import get_parameters, set_parameters, train, test_2_server, trainbic

class BiCClient(BaselineClient):
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr, num_rounds):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
        self.num_rounds = num_rounds
        self.bic_prams = None
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_parameters(self.net, ndarrays_original)

        train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)
        if(ins.config.get("server_round") == self.num_rounds):
            trainbic(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)
        #lay tham so
        modelr_ndarrays = get_parameters(self.net)
        # self.bic_prams = modelr_ndarrays[-1]
        # torch.save(self.bic_prams, bic_path)
        # print(f"[Client {self.partition_id}] Saved BiC Layer to {bic_path}")
        model_ndarrays = modelr_ndarrays[:-1]

        #%
        self.bic_prams = modelr_ndarrays[-1]


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
        #print(f"[DEBUG] ins.parameters: {ins.parameters}")
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        #%
        if self.bic_prams is not None:
            print('Add bic layer in last round')
            ndarrays_original.append(self.bic_prams)
        # #neu dung biclayer de eval thi them dong nay
        # ndarrays_original.append(self.bic_params_client)
        # bic_path = f"bic_layer_client_{self.partition_id}.pt"
        # if self.bic_prams is None and os.path.exists(bic_path):
        #   self.bic_prams = torch.load(bic_path)
        # print(f"[Client {self.partition_id}] Load BiC Layer to {bic_path}")

        # if self.bic_prams is not None:
        #   #set_parameters(self.net, self.bic_prams)
        #   ndarrays_original.append(self.bic_prams)

        # else:
        #   print("[WARNING] self.bic_prams is None! Skipping update for BiC Layer.")
        #   print(self.bic_prams)
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
# class BiCClient(BaselineClient):
#     def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr, bic_params_client):
#         super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
#         self.bic_params_client = bic_params_client

#     def fit(self, ins: FitIns) -> FitRes:
#         print(f"[Client {self.partition_id}] fit, config: {ins.config}")

#         parameters_original = ins.parameters
#         ndarrays_original = parameters_to_ndarrays(parameters_original)

#         ndarrays_original.append(self.bic_params_client)

#         set_parameters(self.net, ndarrays_original)

#         train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)

#         #lay tham so
#         modelr_ndarrays = get_parameters(self.net)
#         # self.bic_prams = modelr_ndarrays[-1]
#         # torch.save(self.bic_prams, bic_path)
#         # print(f"[Client {self.partition_id}] Saved BiC Layer to {bic_path}")
#         model_ndarrays = modelr_ndarrays[:-1]

#         parameters_updated = ndarrays_to_parameters(model_ndarrays)

#         status = Status(code=Code.OK, message="Success")
#         return FitRes(
#             status=status,
#             parameters=parameters_updated,
#             num_examples=len(self.trainloader.dataset),
#             metrics = {},
#         )
    
#     def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
#         print(f"[Client {self.partition_id}] evaluate, config: {ins.config}")
#         #print(f"[DEBUG] ins.parameters: {ins.parameters}")
#         # Deserialize parameters to NumPy ndarray's
#         parameters_original = ins.parameters
#         ndarrays_original = parameters_to_ndarrays(parameters_original)
        
#         # #neu dung biclayer de eval thi them dong nay
#         # ndarrays_original.append(self.bic_params_client)
#         # bic_path = f"bic_layer_client_{self.partition_id}.pt"
#         # if self.bic_prams is None and os.path.exists(bic_path):
#         #   self.bic_prams = torch.load(bic_path)
#         # print(f"[Client {self.partition_id}] Load BiC Layer to {bic_path}")

#         # if self.bic_prams is not None:
#         #   #set_parameters(self.net, self.bic_prams)
#         #   ndarrays_original.append(self.bic_prams)

#         # else:
#         #   print("[WARNING] self.bic_prams is None! Skipping update for BiC Layer.")
#         #   print(self.bic_prams)
#         set_parameters(self.net, ndarrays_original)
#         loss, accuracy, precision, recall, f1_score = test_2_server(self.net, self.valloader)


#         # Build and return response
#         status = Status(code=Code.OK, message="Success")
#         return EvaluateRes(
#             status=status,
#             loss=float(loss),
#             num_examples=len(self.valloader.dataset),
#             metrics={"accuracy": float(accuracy), "cid":self.partition_id, "precision": precision, "recall": recall, "f1_score": f1_score, "loss": loss},
#         )
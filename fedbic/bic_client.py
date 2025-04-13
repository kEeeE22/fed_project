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

'''Ver này train lớp bic sau khi nhận model global.
lr: chỉ train trong round cuối
er: train sau khi nhận global model mỗi round'''
'''mode = lr/er'''
class BiCClient(BaselineClient):
    def __init__(self, partition_id, net, trainloader, valloader, epochs, client_lr, num_rounds, mode):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
        self.mode = mode
        self.num_rounds = num_rounds
        self.bic_prams = None
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        if self.bic_prams is not None:
            ndarrays_original.append(self.bic_prams)
        set_parameters(self.net, ndarrays_original)
        if self.mode == 'er2':
            #chi train lop bic
            print('Train BiC layer')
            trainbic(self.net, self.trainloader, epochs=5, lr=0.001, frozen=True)
        print('Train model')
        if self.mode == 'al':
            #
            trainbic(self.net, self.trainloader, epochs=self.epochs, lr=ins.config.get('client_lr'), frozen=False)
        else:
            train(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=True)
            
        #lay tham so
        modelr_ndarrays = get_parameters(self.net)
        # self.bic_prams = modelr_ndarrays[-1]
        # torch.save(self.bic_prams, bic_path)
        # print(f"[Client {self.partition_id}] Saved BiC Layer to {bic_path}")
        model_ndarrays = modelr_ndarrays[:-1]

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
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)
        #%
        # if ins.config.get("server_round") == self.num_rounds:
        #     print('Add bic layer in last round')
        #     ndarrays_original.append(self.bic_prams)
        if self.mode == 'lr':
            if(ins.config.get("server_round") == self.num_rounds):
                trainbic(self.net, self.trainloader, epochs=200, lr=0.01, frozen=True)
        elif self.mode == 'er1':
            trainbic(self.net, self.trainloader, 1, 0.01, frozen=True)
        
        loss, accuracy, precision, recall, f1_score = test_2_server(self.net, self.valloader)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader.dataset),
            metrics={"accuracy": float(accuracy), "cid":self.partition_id, "precision": precision, "recall": recall, "f1_score": f1_score, "loss": loss},
        )
    
'''2phase fedbic: train model + train bic layer'''
'''mode = FedBic_2phase'''
class TwoPhaseBiCClient(BaselineClient):
    def __init__(self, partition_id, net, trainloader,bic_trainloader, valloader, epochs, client_lr, num_rounds, mode):
        super().__init__(partition_id, net, trainloader, valloader, epochs, client_lr)
        self.mode = mode
        self.num_rounds = num_rounds
        self.bic_trainloader = bic_trainloader
        self.bic_prams = None
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.partition_id}] fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        if self.bic_prams is not None:
            ndarrays_original.append(self.bic_prams)
        set_parameters(self.net, ndarrays_original)
        if self.mode == 'FedBic_2phase':
            trainbic(self.net, self.trainloader, epochs=self.epochs, lr=self.client_lr, frozen=False)
            trainbic(self.net, self.bic_trainloader, epochs=2, lr=0.01, frozen=True)
            
        #lay tham so
        modelr_ndarrays = get_parameters(self.net)
        # self.bic_prams = modelr_ndarrays[-1]
        # torch.save(self.bic_prams, bic_path)
        # print(f"[Client {self.partition_id}] Saved BiC Layer to {bic_path}")
        model_ndarrays = modelr_ndarrays[:-1]

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
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)
        #%
        # if ins.config.get("server_round") == self.num_rounds:
        #     print('Add bic layer in last round')
        #     ndarrays_original.append(self.bic_prams)
        if self.mode == 'lr':
            if(ins.config.get("server_round") == self.num_rounds):
                trainbic(self.net, self.trainloader, epochs=200, lr=0.01, frozen=True)
        elif self.mode == 'er1':
            trainbic(self.net, self.trainloader, 1, 0.01, frozen=True)
        
        loss, accuracy, precision, recall, f1_score = test_2_server(self.net, self.valloader)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader.dataset),
            metrics={"accuracy": float(accuracy), "cid":self.partition_id, "precision": precision, "recall": recall, "f1_score": f1_score, "loss": loss},
        )

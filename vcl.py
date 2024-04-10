import numpy as np
from tqdm import tqdm
import torch

import torchvision
from torch import nn
from torch.optim import Adam, SGD
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, num_epochs, dataloader, task_id, beta, coresets=False):
    
    elbo = ELBO(model, len(dataloader.dataset), beta)

    beta = 0 if coresets else beta
    optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = torch.zeros(inputs.shape[0], 10, 10, device=device)

            for i in range(10):
                net_out = model(inputs, task_id)
                outputs[:, :, i] = F.log_softmax(net_out, dim=-1)

            log_output = torch.logsumexp(outputs, dim=-1) - np.log(10)
            kl = model.get_kl(task_id)
            loss = elbo(log_output, targets, kl)
            loss.backward()
            optimizer.step()



def predict(model, dataloader, task_id):

    model.train()
    accs = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = torch.zeros(inputs.shape[0], 10, 10, device=device)

        for i in range(10):
            with torch.no_grad():
                net_out = model(inputs, task_id)
            outputs[:, :, i] = F.log_softmax(net_out, dim=-1)

        log_output = torch.logsumexp(outputs, dim=-1) - np.log(10)
        accs.append(calculate_accuracy(log_output, targets))
    
    return np.mean(accs)


def run_vcl(num_tasks, single_head, num_epochs, dataloaders, model,
            coreset_method, coreset_size=0, beta=1, update_prior=True):
    
    coreset_list = []
    all_accs = np.empty(shape=(num_tasks, num_tasks))
    all_accs.fill(np.nan)
    for task_id in range(num_tasks):
        print("Running Task", task_id + 1)

        # Train on non-coreset data
        trainloader, testloader = dataloaders[task_id]
        train(model, num_epochs, trainloader, False, task_id, beta)

        # Attach a new coreset
        if coreset_size > 0:
            coreset_method(coreset_list, trainloader, num_samples=coreset_size)

            # coresets old tasks using coresets
            for task in range(task_id + 1):
                print( "Running Coreset", task + 1)
                train(model, num_epochs, coreset_list[task], task, beta, coresets=True)
        print()

        # Evaluate on old tasks
        for task in range(task_id + 1):
            _, testloader_i = dataloaders[task]
            accuracy = predict(model, testloader_i, task)
            print("Task {} Accuracy: {}".format(task + 1, accuracy))
            all_accs[task_id][task] = accuracy
        print()
        if update_prior:
            model.update_prior()
    print(all_accs)
    return all_accs


def run_vanilla_vcl(model, dataloaders, num_tasks, num_epochs, device, beta=0.01):
    coreset_size = 0
    coreset_method = None
    all_accs = run_vcl(num_tasks, False, num_epochs, dataloaders,
                model, coreset_method, coreset_size, beta=beta)
    return all_accs

def run_coresetonly(num_tasks, single_head, num_epochs, dataloaders, model,
            coreset_method, coreset_size=0, beta=1, update_prior=True):
    """Train only on coresets and don't update the prior."""
    update_prior=False
    
    coreset_list = []
    all_accs = np.empty(shape=(num_tasks, num_tasks))
    all_accs.fill(np.nan)
    for task_id in range(num_tasks):
        print("Running Task", task_id + 1)

        # Train on non-coreset data
        trainloader, _ = dataloaders[task_id]

        # Attach a new coreset
        coreset_method(coreset_list, trainloader, num_samples=coreset_size)
        # coresets old tasks using coresets
        #TRAINING ONLY ON CORESETS
        for task in range(task_id + 1):
            print( "Running Coreset", task + 1)
            train(model, num_epochs, coreset_list[task],  task, beta, coresets=True)
            print("Done Training Coreset", task_id+1)

            ### TESTING IMMEDIATELLY AFTER TRAINING THE CORESET
            _, testloader_i = dataloaders[task]
            accuracy = predict(model, testloader_i, task)
            print("Task {} Accuracy: {}".format(task + 1, accuracy))
            all_accs[task_id][task] = accuracy
        print()
        # if update_prior:
        #     model.update_prior()
    print(all_accs)
    return all_accs

class ELBO(nn.Module):

    def __init__(self, model, train_size, beta):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.beta = beta
        self.train_size = train_size

    def forward(self, outputs, targets, kl):
        assert not targets.requires_grad
        # print(F.nll_loss(outputs, targets, reduction='mean'), self.beta * kl / self.num_params)
        return F.nll_loss(outputs, targets, reduction='mean') + self.beta * kl / self.num_params


def calculate_accuracy(outputs, targets):
    return np.mean(outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy())


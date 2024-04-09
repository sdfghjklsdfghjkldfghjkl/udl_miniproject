import torch

import vcl
import dataset
import coresets
import plots
import numpy as np
from models import MultiheadModel

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_tasks = 5
num_epochs = 10 # 100 or 10
single_head = False
batch_size = 256

class_distribution = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]
dataloaders = dataset.SplitMnistDataloader(class_distribution, batch_size)
model = MultiheadModel(28*28)
model.to(device)

# Vanilla VCL - no coreset
coreset_size = 0
coreset_method = coresets.attach_random_coreset_split
vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)

# Random Coreset VCL
coreset_size = 200
coreset_method = coresets.attach_random_coreset_split
vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)

# K-Center Coreset VCL
coreset_size = 40
coreset_method = coresets.attach_kCenter_coreset_split
all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)
accs = np.nanmean(all_accs, axis=1)
print("Average accuracy after each task:", accs)

plots.single_accuracy_plot(accs)

# Random Coreset Only
coreset_size = 200
coreset_method = coresets.attach_random_coreset_split
all_accs = vcl.run_coresetonly(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)
accs = np.nanmean(all_accs, axis=1)
print("Random Only")
print("Average accuracy after each task:", accs)

# K-center Coreset Only
coreset_size = 200
coreset_method = coresets.attach_kCenter_coreset_split
all_accs = vcl.run_coresetonly(num_tasks, single_head, num_epochs, dataloaders,
            model, coreset_method, coreset_size, beta=0.01)
accs = np.nanmean(all_accs, axis=1)
print("Average accuracy after each task:", accs)


#Better batch
# joint_class_distribution = [
#     [0, 1],
#     [0, 1, 2, 3],
#     [0, 1, 2, 3, 4, 5],
#     [0, 1, 2, 3, 4, 5, 6, 7],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# ]
# dataloaders = dataset.SplitMnistDataloader(class_distribution, batch_size)
# coreset_size = 0 #irrelevant
# coreset_method = coresets.attach_kCenter_coreset_split #irrelevant
# all_accs = vcl.run_vcl(num_tasks, single_head, num_epochs, dataloaders,
#             model, coreset_method, coreset_size, beta=0.01)
# accs = np.nanmean(all_accs, axis=1)
# print("Average accuracy after each task:", accs)


# # Batch??? vcl.run_full
# joint_class_distribution = [
#     [0, 1],
#     [0, 1, 2, 3],
#     [0, 1, 2, 3, 4, 5],
#     [0, 1, 2, 3, 4, 5, 6, 7],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# ]
# dataloaders = dataset.SplitMnistDataloader(class_distribution, batch_size)
# coreset_size = 0 #irrelevant
# coreset_method = coresets.attach_kCenter_coreset_split #irrelevant
# all_accs = vcl.run_full(num_tasks, single_head, num_epochs, dataloaders,
#             model, coreset_method, coreset_size, beta=0.01)
# accs = np.nanmean(all_accs, axis=1)
# print("Average accuracy after each task:", accs)



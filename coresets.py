import torch


def attach_random_coreset_split(coresets, sub_train_loader, num_samples=200):
    """
    Attaches a randomly selected subset (coreset) from a given DataLoader to a list of coresets.
    
    This function selects a random subset from `sub_train_loader` based on `num_samples` and creates a new DataLoader for this subset.
    It then appends this new DataLoader to the provided list `coresets`. The original DataLoader (`sub_train_loader`) is modified
    by removing the selected subset indices, effectively partitioning the dataset.
    
    Args:
        coresets (list): A list where the new coreset DataLoader is appended.
        sub_train_loader (DataLoader): The DataLoader from which a random coreset is drawn.
        num_samples (int): The number of samples to include in the coreset.
    """
    
    # Extract current sample indices from the DataLoader
    task_indices = sub_train_loader.sampler.indices
    # Shuffle indices to ensure the randomness of the coreset
    shuffled_task_indices = task_indices[torch.randperm(len(task_indices))]
    # Select a subset of indices to form the coreset
    coreset_indices = shuffled_task_indices[:num_samples]
    # Update the DataLoader to exclude the coreset indices, thus reducing its size
    sub_train_loader.sampler.indices = shuffled_task_indices[num_samples:]
    # Create a new sampler for the coreset based on the selected indices
    coreset_sampler = torch.utils.data.SubsetRandomSampler(coreset_indices)
    # Create a new DataLoader for the coreset
    coreset_loader = torch.utils.data.DataLoader(
        sub_train_loader.dataset, 
        batch_size=sub_train_loader.batch_size, 
        sampler=coreset_sampler)
    # Append the new coreset DataLoader to the list of existing coresets
    coresets.append(coreset_loader)


import torch
############## attempt at faster K-Center implementation
def attach_kCenter_coreset_split(coresets, sub_train_loader, num_samples=200):

    dataset = sub_train_loader.dataset
    task_indices = sub_train_loader.sampler.indices
    loader_size = len(task_indices)

    # Initialization of the first index
    current_index = torch.randint(0, loader_size, (1,)).item()
    coreset_indices = [task_indices[current_index]]
    
    # Convert dataset to a tensor for vectorized operations
    # data_tensor[0] will be the element in dataset with index 0.
    data_tensor = torch.stack([dataset[i][0].flatten().float() for i in task_indices])
    
    # Initialize distances
    distances = torch.full((loader_size,), float('inf'), device=data_tensor.device)
    indices_to_remove = [current_index]
    for _ in range(num_samples - 1):
        # Update distances
        distances = _update_kcenter_distance_vectorized(distances, data_tensor, data_tensor[current_index])
        
        # Find the farthest point
        current_index = distances.argmax().item()
        indices_to_remove.append(current_index)
        coreset_indices.append(task_indices[current_index])

    # Update the DataLoader to exclude the coreset indices, thus reducing its size
    remaining_indices = list(set(range(loader_size)) - set(indices_to_remove))

    sub_train_loader.sampler.indices = task_indices[remaining_indices]

    # Create a new DataLoader for the coreset
    coreset_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=sub_train_loader.batch_size, 
        sampler=torch.utils.data.SubsetRandomSampler(torch.tensor(coreset_indices)))
    
    # Append the new coreset DataLoader to the list of existing coresets
    coresets.append(coreset_loader)

def _update_kcenter_distance_vectorized(distances, data_tensor, current_point):
    # Compute distances from current_point to all points in data_tensor
    new_distances = torch.norm(data_tensor - current_point, dim=1)
    distances = torch.min(distances, new_distances)
    return distances

def attach_qr_coreset_split(coresets, sub_train_loader, num_samples=200):

    dataset = sub_train_loader.dataset
    task_indices = sub_train_loader.sampler.indices
    A = torch.stack([dataset[i][0].flatten().float() for i in task_indices])

    P_indices = qrcp(A)

    coreset_indices = task_indices[P_indices[:num_samples]]

    sub_train_loader.sampler.indices = task_indices[P_indices[num_samples:]]
    
    # Create a new DataLoader for the coreset
    coreset_loader = torch.utils.data.DataLoader(
        sub_train_loader.dataset, 
        batch_size=sub_train_loader.batch_size, 
        sampler=torch.utils.data.SubsetRandomSampler(coreset_indices))
    
    # Append the new coreset DataLoader to the list of existing coresets
    coresets.append(coreset_loader)

def qrcp(B):
    A = B.t() #the transpose
    m, n = A.shape
    Q = torch.eye(m, m, dtype=A.dtype, device=A.device)
    R = A.clone()
    P = torch.arange(n, dtype=torch.long, device=A.device)
    for i in range(min(m, n)):
        # Find the pivot - the column with the largest norm
        norms = torch.norm(R[i:, i:], dim=0)
        if i > 0:
            max_norm_col = torch.argmax(norms) + i
        else:
            max_norm_col = 0
        # Swap columns in R and rows in P
        R[:, [i, max_norm_col]] = R[:, [max_norm_col, i]]
        P[i], P[max_norm_col] = P[max_norm_col].item(), P[i].item()
        # Compute the Householder transformation for the current column
        x = R[i:, i]
        rho = -torch.sign(x[0]) if x[0] != 0 else -1
        s = torch.sqrt(0.5 * (1 + abs(rho) * x[0] / torch.norm(x)))
        v = torch.zeros_like(x)
        v[0] = s
        u = (x - rho * v) / (s * x[0])
        v = u - v
        # Apply the transformation to R and Q
        R[i:, i:] -= 2 * v.outer(v @ R[i:, i:])
        Q[:, i:] -= 2 * (Q[:, i:] @ v).outer(v)
    return P


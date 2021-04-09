import torch
from torch.utils.data import DataLoader

import paths
from datasets.dataset import BasicDataset

paths=paths.Paths(db="SegTHOR")
dataset = BasicDataset(paths=paths, scale=1)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

mean = 0
std = 0
t=0
stds = []

for batch in loader:
    imgs = batch['image']
    batch_samples = 1  # batch size (the last batch can have smaller size!)
    curr_mean = torch.mean(imgs)
    mean += curr_mean
    stds.append(curr_mean)

mean /= len(dataset)
print(f"Mean: {mean}")

for m in stds:
    curr_t = pow((m - mean),2)
    t += curr_t

std = t / len(stds)

print(f"Std deviation: {std}")

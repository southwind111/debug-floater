import torch
a = torch.tensor([2,3,4,5,6])
b = torch.tensor([1,2,4])

a[b] += 1
print(a)
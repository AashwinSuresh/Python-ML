import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
device = torch.device("cuda")
print(device)
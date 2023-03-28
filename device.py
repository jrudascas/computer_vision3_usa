
import torch
print("Cuda is available, ", torch.cuda.is_available())

print("Current device, ", torch.cuda.current_device())

print("Device(0), ", torch.cuda.device(0))

print("Device count, ", torch.cuda.device_count())

print("Device name, ", torch.cuda.get_device_name(0))

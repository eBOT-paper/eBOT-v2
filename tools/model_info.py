from ebot.models import ConvNet, SimpleCNN, ResNet18, ResNet52
import torchvision.models as models

model = ResNet52()
#model = models.resnet50()
# 1. Count the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

# 2. Count only trainable parameters (optional)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# 3. Estimate model size in bytes
model_size_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
model_size_MB = model_size_bytes / (1024 ** 2)
print(f"Model size: {model_size_MB:.2f} MB")

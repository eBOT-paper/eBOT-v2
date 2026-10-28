import os
import json
import io


def load_config(fname: str) -> dict:
    with open(fname, "r") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content

def evaluate(model, test_loader):
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
    return 100.0 * correct / total

def prepare_data(data_type: str, rank: int, world_size: int):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Dataset
    from filelock import FileLock
    
    with FileLock(os.path.expanduser("~/data.lock")):
        batch_size = 16
        if data_type == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            full_dataset = datasets.MNIST("~/data", train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST("~/data", train=False, transform=transform)
        elif data_type == "cifar10":
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            full_dataset = datasets.CIFAR10("~/data", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10("~/data", train=False, transform=transform)
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank
        )
        dataloader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False
        )
        testloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    
    return dataloader, testloader

def print_model_info(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    model_size_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    model_size_MB = model_size_bytes / (1024 ** 2)
    print(f"Model size: {model_size_MB:.2f} MB")

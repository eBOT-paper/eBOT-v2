import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

from ebot.utils import evaluate, load_config, prepare_data
from ebot.models import ConvNet, SimpleCNN, ResNet18, ResNet52
from ebot.workers import EbpfWorker, TorchDDPWorker, TorchTCPWorker

load_dotenv()

master_ip = os.getenv("COORD_IP")
master_port = 29500

config = load_config("configs/local_config.json")

rank = config["id"]
world_size = config["worker_num"]
model_type = config["model_type"]
learning_rate = config["learning_rate"]

worker_type = sys.argv[1]


if __name__ == "__main__":
    torch.manual_seed(42 + rank)

    MODEL_CLASSES = {
        "convnet": ConvNet,
        "resnet18": ResNet18,
        "resnet52": ResNet52
    }
    DATASETS = {
        "convnet": "mnist",
        "resnet18": "cifar10",
        "resnet52": "cifar10"
    }
    WORKER_CLASSES = {
        "ebpf": EbpfWorker,
        "torch_ddp": TorchDDPWorker,
        "torch_tcp": TorchTCPWorker
    }

    model_class = MODEL_CLASSES[model_type]()
    dataloader, testloader = prepare_data(DATASETS[model_type], rank, world_size)
    worker = WORKER_CLASSES[worker_type](model_class)

    device = worker.setup(rank, world_size, master_ip, master_port)
    
    model = worker.model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print("Running synchronous parameter server training.")

    epochs = 1
    for epoch in range(epochs):
        total_loss = 0.0
        total_time = 0.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            start_time = time.perf_counter()
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            model.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()

            worker.aggregate(batch_idx)

            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)

            if rank == 0 and batch_idx > 0:
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                avg_time = total_time / (len(dataloader) - 1)
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{batch_idx}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Elapsed time: {elapsed:.6f}s"
                 )

        if rank == 0:
            accuracy = evaluate(model, testloader)
            print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Average Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.6f}, "
                    f"Avg. Batch Time: {avg_time:.6f}s"
                 )

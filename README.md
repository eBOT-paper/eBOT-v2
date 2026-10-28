# Distributed Training with eBPF

## Note

This setup has been tested on **Ubuntu 22.04.3 LTS** with **Linux 5.15.0-125-generic**. Other environments may require adjustments.

---

## Project Structure

```
.
├── agent.py              # Agent running on each node (receives config/commands from the server)
├── server.py             # Coordinator server
├── worker.py             # Training entry point (eBPF / Torch DDP / Torch TCP)
├── ebot/                 # Python package
│   ├── models.py         #   Model definitions
│   ├── utils.py          #   Config loading, data preparation, evaluation
│   └── workers/          #   Worker implementations per communication backend
├── ebpf/                 # eBPF programs (XDP/TC), user-space loader and shared map library
├── configs/              # main_config.json (+ generated system_config.json / local_config.json)
├── tools/                # Helper scripts (system config generation, SSH config, model info)
├── scripts/              # Shell scripts for setup, running and cleaning eBPF programs
├── simulation/           # Aggregation-time simulation, solver, results and plots
└── docs/                 # Appendix
```

All commands below are run from the repository root.

---

## Automatic Setup

On each node, run the following command to install all required dependencies for running eBPF programs:

```bash
./scripts/setup.sh
```

---

## Python Dependencies

If Python 3 is not installed, we recommend using **Python 3.9**.
The **server-agent** requires the `websockets` package for communication. Install it as follows:

```bash
sudo apt install python3-pip
pip3 install websockets
```

---

## Topology Configuration

1. Edit the node description in `configs/main_config.json`.
2. Generate the system configuration file with:

   ```bash
   python3 -m tools.gen_system_config
   ```

   This will create or update `configs/system_config.json` and `.env`.

---

## Configuration

### Coordinator Node

1. Edit the configuration file `configs/main_config.json` to assign communication jobs.
2. Start the server with:

   ```bash
   python3 server.py
   ```

### Worker Nodes

Run the agent process on each node with:

```bash
python3 agent.py
```

Once all nodes are connected, select **1. Configuration** from the coordinator to automatically propagate the configuration across all nodes.

---

## Running eBPF Programs

There are two options to start the eBPF programs:

* **Automated**: Use the **server-agent** system to launch programs on all nodes.
* **Manual**: On each node, run:

  ```bash
  ./scripts/run.sh
  ```

---

## Running Training Programs

After the eBPF programs are active on all nodes, launch training jobs:

* **Manual**: Run on each node:

  ```bash
  python3 worker.py
  ```
* **Automated**: Use the **server-agent** system to start training across all nodes.

---

## Automatically Starting Agents

Run the following steps as root:

1. Ensure `nohup` is available:

   ```bash
   apt install coreutils
   ```
2. Start all agents:

   ```bash
   ./scripts/agents.sh
   ```


import asyncio
import aioconsole
import websockets
import json
import os
from dotenv import load_dotenv
from ebot.utils import load_config

load_dotenv()

COORD_IP = os.getenv("COORD_IP")
COORD_PORT = os.getenv("COORD_PORT")

config = load_config("configs/system_config.json")
clients = set()


async def handle_client(websocket):
    clients.add(websocket)
    print(f"\nNew client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"\n\nClient {websocket.remote_address}: {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"\nClient {websocket.remote_address} disconnected.")
        clients.remove(websocket)
 

async def _send_to_clients(event_type: str, data: any, targets: list):
    if not targets:
        return

    event = json.dumps({"event": event_type, "data": data})
    await asyncio.gather(*(client.send(event) for client in targets))


async def broadcast(event_type: str, data: any = ""):
    await _send_to_clients(event_type, data, clients)


async def multicast(event_type: str, data: any = ""):
    config = load_config("configs/system_config.json")
    allowed_addrs = set(config.get("clients", {}).keys())
    
    targets = [
        client for client in clients
        if client.remote_address[0] in allowed_addrs
    ]
    await _send_to_clients(event_type, data, targets)


async def update_local_config():
    config = load_config("configs/system_config.json")
    train_cfg = config.get("train_config", {})
    
    tasks = []
    for client in clients:
        addr = client.remote_address[0]
        client_cfg = config.get("clients", {}).get(addr, {})
        
        if client_cfg:
            update_data = {
                **client_cfg,
                **train_cfg,
                "ip": addr,
            }
            event = json.dumps({"event": "update_local_config", "data": update_data})
            task = asyncio.create_task(client.send(event))
            tasks.append(task)
    
    if tasks:
        await asyncio.gather(*tasks)


async def run_ebpf_progs():
    await multicast("run_ebpf_progs")

async def run_train_progs(prog_type: str):
    await multicast(f"run_{prog_type}_train")

async def clean_progs():
    await broadcast("clean_progs")

async def ping():
    if not clients:
        print("\nNo clients connected.")
    else:
        print(f"\n{len(clients)} clients connected.")
    await broadcast("ping", "Ping!")

async def kill_agents():
    await broadcast("kill_agent")

async def event_dispatcher():
    while True:
        print("\nSelect an option:")
        print("1. Update local config")
        print("2. Run eBPF programs")
        print("3. Run eBPF training")
        print("4. Run Torch DDP training")
        print("5. Run Torch TCP training")
        print("6. Clean programs")
        print("7. Ping")
        print("8. Kill agents")
        
        choice = (await aioconsole.ainput("Enter your choice (1->8): ")).strip()
        
        if choice == "1":
            await update_local_config()
        elif choice == "2":
            await run_ebpf_progs()
        elif choice == "3":
            await run_train_progs("ebpf")
        elif choice == "4":
            await run_train_progs("torch_ddp")
        elif choice == "5":
            await run_train_progs("torch_tcp")
        elif choice == "6":
            await clean_progs()
        elif choice == "7":
            await ping()
        elif choice == "8":
            await kill_agents()
        else:
            continue
        
async def main():
    server = await websockets.serve(handle_client, COORD_IP, COORD_PORT)
    print(f"\nServer started on ws://{COORD_IP}:{COORD_PORT}")
    await asyncio.gather(server.wait_closed(), event_dispatcher())

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nServer terminated manually.")



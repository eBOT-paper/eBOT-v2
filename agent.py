import os
import time
import json
import asyncio
import websockets
import subprocess
import jinja2
import ipaddress
from dotenv import load_dotenv
from ebot.utils import load_config

load_dotenv()

COORD_IP = os.getenv("COORD_IP")
COORD_PORT = os.getenv("COORD_PORT")


def ipv4_to_hex(addr: str) -> str:
    return f"0x{int(ipaddress.IPv4Address(addr)):08X}"


def parse_mac(mac: str) -> str:
    return ",".join(["0x"+seg for seg in mac.split(":")])


def write_local_config(data: dict, config_path: str = "configs/local_config.json"):
    content = json.dumps(data, indent=4)
    with open(config_path, "w") as f:
        f.write(content)


def write_agg_config(
    data: dict,
    template_path: str = "ebpf/agg_common.h.tmpl", 
    output_path: str = "ebpf/agg_common.h"
): 
    input_data = {
        "dummy_ip": ipv4_to_hex(data["dummy_ip"]),
        "port": data["port"],
        "host_id": data["id"],
        "host_ip": ipv4_to_hex(data["ip"]),
        "host_mac": parse_mac(data["mac"]),
        "worker_num": data["worker_num"],
        "scale_factor": data["scale_factor"],
        "gradient_size": data["gradient_size"],
        "fragment_size": data["fragment_size"],
    }
    
    ids = [str(child["id"]) for child in data.get("children", [])]
    addrs = [ipv4_to_hex(child["addr"]) for child in data.get("children", [])]
    macs = [parse_mac(child["mac"]) for child in data.get("children", [])]

    input_data.update({
        "children_num": len(ids),
        "children_id": ",".join(ids),
        "children_ip": ",".join(addrs),
        "children_mac": ",".join(macs),
    })

    parent = data.get("parent")
    if parent:
        input_data.update({
        "parent_id": parent["id"],
        "parent_ip": ipv4_to_hex(parent["addr"]),
        "parent_mac": parse_mac(parent["mac"]),
        "parent_num": 1,
    })
    else:
        input_data["parent_num"] = 0
    
    with open(template_path, "r") as f:
        file_tmpl = f.read()
    
    template = jinja2.Template(file_tmpl)
    file_content = template.render(**input_data)
    
    with open(output_path, "w") as f:
        f.write(file_content)


async def update_local_config(data: dict):
    write_local_config(data)
    write_agg_config(data)


async def run_ebpf_progs():
    try:       
        subprocess.Popen("./scripts/run.sh", shell=True)
        subprocess.Popen("python3 worker.py", shell=True)
    except Exception as e:
        print(f"Error running program: {e}")

async def run_train_progs(prog_type: str):
    config = load_config("configs/local_config.json")
    if not config['children']:
        time.sleep(5)
    try:       
        subprocess.Popen(f"python3 worker.py {prog_type}", shell=True)
    except Exception as e:
        print(f"Error running program: {e}")


async def clean_progs():
    try:
        subprocess.run("./scripts/clean.sh", shell=True, check=True) 
    except Exception as e:
        print(f"Error running program: {e}")

async def handle_server_messages(websocket):
    try:
        async for message in websocket:
            event = json.loads(message)
            event_type = event.get("event")
            data = event.get("data")

            print(f"Received event: {event_type}, data: {data}")

            if event_type == "update_local_config":
                await update_local_config(data)
                banner = "Updated!"
                await websocket.send(banner)
                print(banner)
            
            elif event_type == "run_ebpf_progs":
                await run_ebpf_progs()
                banner = "Run ebpf!"
                await websocket.send(banner)
                print(banner)
            
            elif event_type == "run_ebpf_train":
                await run_train_progs("ebpf")
                banner = "Run ebpf training!"
                await websocket.send(banner)
                print(banner)
            
            elif event_type == "run_torch_ddp_train":
                await run_train_progs("torch_ddp")
                banner = "Run torch ddp training!"
                await websocket.send(banner)
                print(banner)
            
            elif event_type == "run_torch_tcp_train":
                await run_train_progs("torch_tcp")
                banner = "Run torch tcp training!"
                await websocket.send(banner)
                print(banner)
            
            elif event_type == "clean_progs":
                await clean_progs()
                await websocket.send("Cleaned!")
                print("Cleaned!")
            
            elif event_type == "ping":
                print("Sending back to server: Pong!")
                await websocket.send("Pong!")
            
            elif event_type == "kill_agent":
                print("Client is shutting down.")
                return False
            
            elif event_type == "shutdown":
                print("Server is shutting down. Exiting client.")
                return False
        
        return True

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server. Exiting client.")
        return False
    

async def main():
    try:
        async with websockets.connect(f"ws://{COORD_IP}:{COORD_PORT}") as websocket:
            print("Connected to server.")
            while True:
                is_running = await handle_server_messages(websocket)
                if not is_running:
                    break
            
                await asyncio.sleep(20)
                try:
                    await websocket.ping()
                except:
                    print("Server did not respond to ping. Exiting client.")
                    break
    except (websockets.exceptions.ConnectionClosedError,
            ConnectionRefusedError):
        print("Could not connect to server. Exiting.")


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Client terminated manually.")



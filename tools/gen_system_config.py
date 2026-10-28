import json
from typing import Optional
from ebot.utils import load_config


def load_nodes(config_path: str) -> tuple:
    config = load_config(config_path)
    nodes = [node for node in config["node_config"] if not node.get("disabled")]
    config["train_config"]["worker_num"] = len(nodes)
    return config, nodes


G = 1000.0
TAU = 0.1
DEFAULT_BANDWIDTH = 10.0


def run_ebot_algorithm(nodes: list) -> dict:
    B = {node["id"]: float(node.get("bandwidth", DEFAULT_BANDWIDTH)) for node in nodes}
    order = sorted(B, key=lambda i: B[i], reverse=True)

    graph = {i: [] for i in order}
    if not order:
        return graph

    degrees = {i: 0 for i in order}
    depths = {i: 0 for i in order}
    selected = [order[0]]

    for i in order[1:]:
        best_parent = None
        best_cost = float("inf")

        for j in selected:
            temp_degrees = dict(degrees)
            temp_degrees[i] += 1
            temp_degrees[j] += 1

            beta = min(B[k] / (temp_degrees[k] + 1) for k in order)
            D = max(max(depths[k] for k in selected), depths[j] + 1)

            cost = G / beta + TAU * D
            if cost < best_cost:
                best_cost = cost
                best_parent = j

        graph[best_parent].append(i)
        degrees[i] += 1
        degrees[best_parent] += 1
        depths[i] = depths[best_parent] + 1
        selected.append(i)

    return graph

def build_clients(nodes: list, graph: dict) -> dict:

    def get_node(node_id: int) -> Optional[dict]:
        return next((node for node in nodes if node["id"] == node_id), None)

    clients = {}
    for u, children in graph.items():
        node_u = get_node(u)
        if not node_u:
            continue

        addr_u = node_u["addr"]
        clients.setdefault(addr_u, {
            "id": node_u["id"],
            "mac": node_u["mac"],
            "parent": None,
            "children": []
        })

        for v in children:
            node_v = get_node(v)
            if not node_v:
                continue

            addr_v = node_v["addr"]
            clients[addr_u]["children"].append(node_v)

            if addr_v not in clients:
                clients[addr_v] = {
                    "id": node_v["id"],
                    "mac": node_v["mac"],
                    "parent": node_u,
                    "children": []
                }
            else:
                clients[addr_v]["parent"] = node_u

    return clients


def save_configs(config: dict, clients: dict):
    config["clients"] = clients
    del config["node_config"]

    with open("configs/system_config.json", "w") as f:
        f.write(json.dumps(config, indent=4))

    coord_ip = config["coord_config"]["ip"]
    coord_port = config["coord_config"]["port"]
    env_content = f"COORD_IP={coord_ip}\nCOORD_PORT={coord_port}\n"

    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)

    del config["coord_config"]


def main():
    config, nodes = load_nodes("configs/main_config.json")
    graph = run_ebot_algorithm(nodes)
    print("Generated graph:", graph)

    clients = build_clients(nodes, graph)
    save_configs(config, clients)


if __name__ == "__main__":
    main()

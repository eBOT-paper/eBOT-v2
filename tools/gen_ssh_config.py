import json

with open("./nodes.json", "r") as f:
    content = f.read()
    content = json.loads(content)

nodes = content["nodes"]
data = ""
for node in nodes:
    host_id = int(node["id"])
    data += f"Host node{host_id+1}\n"
    data += f"    HostName {node['addr']}\n"
    data += f"    User ubuntu\n"
    data += f"    Port 22\n\n"

with open("/root/.ssh/config", "w") as f:
    f.write(data)

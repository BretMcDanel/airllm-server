# AirLLM Remote LLM Server
**A private, self‑hosted, low‑VRAM‑friendly LLM server using AirLLM**

This project provides a local, privacy‑preserving LLM inference server powered by [AirLLM](https://github.com/lyogavin/airllm), designed to run efficiently on consumer GPUs with limited VRAM. It exposes a fully OpenAI‑compatible API, allowing any OpenAI‑style client or web UI to connect without modification.

**This project stands on the shoulders of giants.  Without AirLLM it would not exist.  I would like to give a special thanks to Gavin Li for creating and releasing AirLLM.**

The server is ideal for users who want:
* Full control over their AI infrastructure
* Local inference without sending data to third‑party clouds
* Efficient model execution on modest hardware
* A backend for custom agents, automation, or private applications

## Features
* OpenAI‑compatible API (/v1/chat/completions)
* Local‑first, privacy‑first — no external calls unless you choose to add them
* Optimized for low‑VRAM GPUs using AirLLM:
  * 4‑bit quantization
  * Flash attention
  * Memory‑efficient inference
* Remote‑accessible for multi‑machine setups
* Supports any HuggingFace model compatible with AirLLM
* Simple FastAPI server that’s easy to extend
* Optional systemd service for production deployments

## Requirements
### Hardware
* NVIDIA GPU (6–24 GB VRAM recommended)
* CUDA‑compatible driver (minimum version 555.xx)
* 16+ GB system RAM recommended

### Software
* Ubuntu 22.04 / 24.04 (recommended) or Debian 12
* Can run in WSL under Windows
* Install script installs everything else

## Installation
### WSL preparation
Install NVidia drivers as normal in Windows.
#### Install WSL (if it is not already installed)
```powershell
wsl --install
```
Reboot

#### Install new Ubuntu 24.04 image
```powershell
wsl --install Ubuntu-24.04
```

#### Launch WSL (Not needed for first install)
```powershell
wsl -d Ubuntu-24.04
```

### Bare Metal Linux preparation
Install NVIDIA drivers as normal  
Reboot as required

### Install LLM Server
From a linux shell prompt
```bash
git clone https://github.com/BretMcDanel/airllm-server.git
cd airllm-server
./setup.sh

```

## Model Configuration
The server loads a model via AirLLM:

```python
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
```
You can replace this with any supported model, such as:
* meta-llama/Meta-Llama-3-8B-Instruct
* Qwen/Qwen2.5-7B-Instruct
* mistralai/Mistral-7B-Instruct-v0.3
* meta-llama/Meta-Llama-3-70B-Instruct (if VRAM allows)

AirLLM automatically applies memory‑saving optimizations.

## Running the Server
Start the server:
```bash
source .venv/bin/activate
python server.py
```

The API will be available at:

```Code
http://0.0.0.0:8000/v1/chat/completions
```

## Testing the API
### Set API key and URL
```bash
AIRLLM_API_KEY="mysecret"
AIRLLM_URL="http://localhost:8000"

```

#### Using curl:

```bash
curl "$AIRLLM_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AIRLLM_API_KEY" \
  -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "stream": false
      }'

```

#### Using Python:
```python
import os
import requests

API_KEY = os.getenv("AIRLLM_API_KEY")
URL = os.getenv("AIRLLM_URL") + "/v1/chat/completions"

payload = {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain quantum tunneling."}],
    "stream": False
}

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

resp = requests.post(URL, json=payload, headers=headers)
print(resp.json())
```

#### Using Node.js
```js
import fetch from "node-fetch";

const API_KEY = process.env.AIRLLM_API_KEY;
const BASE_URL = process.env.AIRLLM_URL;

const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${API_KEY}`
  },
  body: JSON.stringify({
    model: "meta-llama/Meta-Llama-3-8B-Instruct",
    messages: [{ role: "user", content: "Write a haiku about winter" }],
    stream: false
  })
});

const data = await response.json();
console.log(data);
```

## Running as a Systemd Service (optional)
Create:
```Code
/etc/systemd/system/airllm.service
```
Add:
```Code
[Unit]
Description=AirLLM Server
After=network.target

[Service]
User=<yourusername>
WorkingDirectory=/home/<yourusername>/airllm-server
ExecStart=/home/<yourusername>/airllm-server/.venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```
Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now airllm
```

## Network Access
To allow remote clients (e.g., WSL, another Linux machine, or a web UI):
* Ensure port 8000 is open on your firewall
* Use your server’s LAN IP:

```Code
http://<gpu-server-ip>:8000/v1/chat/completions
```

## Integrating With Clients
Any OpenAI‑compatible client can connect by setting:

```Code
OPENAI_API_BASE=http://<gpu-server-ip>:8000/v1
OPENAI_API_KEY=dummy
```

Examples:
* Web UIs
* Custom agents
* CLI tools
* Python scripts
* Local automation workflows

# LLM Task Automation Agent using LoRA Fine-Tuning

> A lightweight AI agent that interprets natural language instructions, predicts the correct system command using a LoRA fine-tuned LLM, executes it, and returns the output — all running on CPU with just 8GB RAM.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Fine-Tuning with LoRA](#fine-tuning-with-lora)
- [Supported Instructions](#supported-instructions)
- [Memory Requirements](#memory-requirements)
- [Difference from Other Projects](#difference-from-other-projects)
- [Future Improvements](#future-improvements)
- [Interview Tip](#interview-tip)
- [License](#license)

---

## Overview

This project demonstrates how a pre-trained Large Language Model (LLM) can be fine-tuned using **LoRA (Low-Rank Adaptation)** to understand natural language task instructions and map them to real system commands — which are then executed by a Python AI agent.

The agent bridges the gap between **human language** and **machine execution**, showing how LLMs can be used to automate OS-level tasks without writing code manually.

**Designed to run entirely on Google Colab (free tier) with CPU and 8GB RAM.**

---

## Demo

```
You: check disk usage

=======================================================
👤 USER: check disk usage
=======================================================
🤖 PREDICTED COMMAND [lookup table]: df -h

⚙️  Executing...

📋 OUTPUT:
----------------------------------------
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1        50G   20G   30G  40% /
tmpfs           3.9G     0  3.9G   0% /dev/shm
----------------------------------------
=======================================================
```

```
You: show running processes

=======================================================
👤 USER: show running processes
=======================================================
🤖 PREDICTED COMMAND [lookup table]: ps aux --sort=-%cpu | head -15

⚙️  Executing...

📋 OUTPUT:
----------------------------------------
USER       PID %CPU %MEM    VSZ   RSS COMMAND
root         1  0.0  0.1  22560  1024 /sbin/init
root       234  0.2  0.5  45000  4096 python3 agent.py
----------------------------------------
=======================================================
```

---

## Architecture

```
User Instruction
        ↓
Python AI Agent
        ↓
Fine-Tuned LLM (LoRA)
        ↓
Task Classification
        ↓
Tool Selection
        ↓
Execute System Command (subprocess)
        ↓
Return Output to User
```

---

## Features

- 🧠 **LoRA Fine-Tuned LLM** — Trained to map natural language → system commands
- ⚡ **Fast Lookup + LLM Fallback** — Instant response for known commands, LLM handles unknown ones
- 🛠️ **Real Command Execution** — Uses Python `subprocess` to run live OS commands
- 💬 **Natural Language Interface** — No need to remember command syntax
- 🪶 **Ultra Lightweight** — `facebook/opt-125m` model, ~250MB, CPU-only
- ☁️ **Google Colab Ready** — Runs on free tier with no GPU required
- 🔁 **Interactive Mode** — Chat continuously with the agent

---

## Tech Stack

| Category | Tool / Library |
|---|---|
| Language | Python 3 |
| Base Model | `facebook/opt-125m` |
| Fine-Tuning Method | LoRA (PEFT) |
| LLM Framework | HuggingFace Transformers |
| Dataset | Custom JSON instruction-command pairs |
| Command Execution | Python `subprocess` |
| Platform | Google Colab (CPU, Free Tier) |

---

## Project Structure

```
llm-task-automation-agent/
│
├── task_automation_agent.ipynb   # Main Colab notebook (all-in-one)
├── dataset.json                  # Instruction → command training data
├── task_agent_model/             # Saved LoRA fine-tuned model (after training)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Getting Started

### Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the notebook or paste cells manually
3. Set Runtime to **CPU** (no GPU needed)
4. Run all cells in order — done in ~5 minutes

### Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/llm-task-automation-agent.git
cd llm-task-automation-agent

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook task_automation_agent.ipynb
```

### requirements.txt

```
transformers
peft
datasets
accelerate
torch
```

---

## How It Works

### Step 1 — Dataset Creation
A JSON dataset of natural language instruction to system command pairs is created covering 20 common OS tasks.

### Step 2 — LoRA Fine-Tuning
The `facebook/opt-125m` base model is loaded and LoRA adapters are injected into the attention layers. Only the adapter weights are trained, keeping memory usage low. The model learns to predict the correct system command given a natural language instruction.

### Step 3 — AI Agent Execution
When the user types an instruction:

1. **Lookup table** checks for a direct keyword match (fast path)
2. **LLM fallback** handles unrecognized instructions via model inference
3. The resolved command is executed via `subprocess`
4. Output is returned to the user

This two-layer approach makes the agent both **reliable** and **flexible**.

---

## Dataset

Training data is stored in `dataset.json` as instruction-output pairs:

```json
[
  { "instruction": "list files",             "output": "ls" },
  { "instruction": "check disk usage",       "output": "df -h" },
  { "instruction": "show running processes", "output": "ps aux" },
  { "instruction": "check memory usage",     "output": "free -h" },
  { "instruction": "check network ports",    "output": "ss -tuln" },
  { "instruction": "show system info",       "output": "uname -a" }
]
```

**20 instruction-command pairs covering:**
- File system operations
- Process and memory monitoring
- Network and port inspection
- User and session info
- System and OS details
- Disk and partition info
- Environment and cron jobs

---

## Fine-Tuning with LoRA

| Parameter | Value | Reason |
|---|---|---|
| Base Model | `facebook/opt-125m` | ~250MB, CPU-friendly |
| LoRA Rank (`r`) | 8 | Balanced capacity vs memory |
| LoRA Alpha | 32 | Standard scaling factor |
| Target Modules | `q_proj`, `v_proj` | Attention layers only |
| Dropout | 0.05 | Light regularization |
| Max Sequence Length | 128 tokens | Commands are short |
| Batch Size | 2 + grad accum 4 | Simulates batch of 8 |
| Epochs | 5 | Small dataset needs more passes |
| Decoding | Greedy (`do_sample=False`) | Deterministic command output |
| `fp16` | Disabled | CPU does not support fp16 |

LoRA freezes the base model weights and only trains small low-rank adapter matrices — making fine-tuning possible without a GPU.

---

## Supported Instructions

| Natural Language Instruction | Command Executed |
|---|---|
| list files | `ls` |
| list all files / hidden files | `ls -la` |
| check disk usage / disk space | `df -h` |
| show running processes | `ps aux --sort=-%cpu` |
| check memory / RAM | `free -h` |
| check network ports / open ports | `ss -tuln` |
| show current directory | `pwd` |
| check cpu info | `cat /proc/cpuinfo` |
| show uptime | `uptime` |
| check logged in users | `who` |
| show environment variables | `printenv` |
| check os version / system info | `uname -a` |
| show network interfaces | `ip addr` |
| check active connections | `ss -tupn` |
| show cron jobs | `crontab -l` |
| check disk partitions | `lsblk` |
| show file permissions | `ls -la /tmp` |
| top processes | `ps aux --sort=-%cpu \| head -10` |
| list running services | `systemctl list-units` |
| check kernel version | `uname -r` |

---

## Memory Requirements

| Resource | Requirement |
|---|---|
| RAM | 8GB minimum |
| GPU | Not required |
| Disk Space | ~500MB |
| Python Version | 3.8+ |
| Platform | Google Colab (free) or local CPU |

---

## Difference from Other Projects

| Feature | Task Automation Agent | Cybersecurity Agent |
|---|---|---|
| Goal | Automate OS tasks | Security analysis |
| LLM Role | Predict system command | Analyze tool output |
| Dataset | Instruction → command pairs | CVE, logs, vulnerabilities |
| Output | Raw command result | Security recommendations |
| Target Users | Developers, sysadmins | Red teamers, analysts |
| Decoding | Greedy (deterministic) | Sampling (creative) |

---

## Future Improvements

- [ ] Support for Windows commands (`dir`, `tasklist`, `ipconfig`)
- [ ] Add more complex multi-step task chains
- [ ] Gradio / Streamlit web UI
- [ ] Upgrade to TinyLlama-1.1B when GPU is available
- [ ] Voice input support
- [ ] Log all executed commands to a file
- [ ] Add confirmation prompt before running destructive commands
- [ ] Integrate with file management (create, delete, move files)

---


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Built as a portfolio project demonstrating LLM fine-tuning and AI agent design.
Star this repo if it helped you!

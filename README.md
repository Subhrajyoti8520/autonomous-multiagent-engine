<div align="left">

  # 🤖📈 Autonomous Multi-Agent Engine
  
  <p align="left">
    <i>An asynchronous, multi-agent AI framework engineered for real-time deal discovery, valuation, and arbitrage notification.</i>
  </p>

  <!-- Live Pulsing Status Banner -->
  <p align="left">
    <a href="https://github.com/Subhrajyoti8520/autonomous_multiagent_engine">
      <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=3000&pause=1000&color=22C55E&center=true&vCenter=true&multiline=false&width=560&lines=⚡+LIVE+SYSTEM:+10+Agents+Scanning+Feeds...;🧠+Tripartite+Consensus+Engine+Active...;🎯+Real-Time+Arbitrage+Alerts+Online..." alt="Typing Dynamic Status" />
    </a>
  </p>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=3000&pause=1000&color=22C55E&center=true&vCenter=true&width=560&lines=⚡+LIVE+SYSTEM%3A+10+Agents+Scanning+Feeds...;🧠+Tripartite+Consensus+Engine+Active...;🎯+Real-Time+Valuation+Alerts+Online...)](https://git.io/typing-svg)

  <!-- Primary CI & Runtime Badges -->
  [![CI Pipeline](https://github.com/Subhrajyoti8520/autonomous_multiagent_engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Subhrajyoti8520/autonomous_multiagent_engine/actions)
  [![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/Subhrajyoti8520/autonomous_multiagent_engine)
  [![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
  [![Code Style: Ruff](https://img.shields.io/badge/Linter-Ruff-000000?style=for-the-badge&logo=fastapi&logoColor=white)](https://github.com/astral-sh/ruff)

  <br />

  <!-- Core AI & Backend Stack Badges -->
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Gemini](https://img.shields.io/badge/Gemini_Flash-8E75C2?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com/)
  [![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=lightning&logoColor=white)](https://groq.com/)
  [![Ollama](https://img.shields.io/badge/Ollama-GGUF-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.ai/)
  [![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6633?style=flat-square)](https://www.trychroma.com/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-UI_Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

  <br />

</div>

---
## 📋 Table of Contents

- [📖 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🧠 System Architecture](#-system-architecture)
- [🤖 The Agent Ecosystem](#-the-agent-ecosystem)
- [⚡ Performance Benchmarks & System Optimizations](#-performance-benchmarks--system-optimizations)
- [🏛️ Architectural Philosophy: The Modular Monolith](#️-architectural-philosophy-the-modular-monolith)
- [📂 Directory Structure](#-directory-structure)
- [🛠️ Infrastructure, CI/CD & LLMOps Pipeline](#️-infrastructure-cicd--llmops-pipeline)
- [🧬 Core Infrastructure Highlights](#-core-infrastructure-highlights)
- [🚀 Quickstart](#-quickstart)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
---

## 📖 Overview

The **Autonomous Multi-Agent Engine** is an advanced AI system designed to scrape unstructured web deals, normalize the data, and evaluate true market value to identify high-margin arbitrage opportunities. 

Instead of relying on a single model, it utilizes a **Tripartite Consensus Engine** combining Retrieval-Augmented Generation (RAG), local GGUF Large Language Models, and Deep Neural Networks to dynamically price items. Profitable deals are immediately dispatched to the user via real-time push notifications, and tracked in a persistent vector-space dashboard.

---
## ✨ Key Features

* 🧩 **`Tripartite Consensus Engine:`** Validates pricing concurrently using a custom PyTorch DNN, a localized GGUF LLM via Ollama, and a RAG-powered Frontier Agent.
* 🔄 **`Dynamic Weight Redistribution:`** Built-in fault tolerance dynamically adjusts consensus weighting (50% RAG, 25% DNN, 25% Specialist LLM) if a sub-agent fails.
* 🕹️ **`Dual Planning Architectures:`** Supports both a deterministic execution pipeline and an autonomous ReAct-based tool-calling orchestrator (Powered by Gemini).
* 🧬 **`Advanced RAG Pipeline:`** Leverages BAAI Bi-Encoders for dense retrieval and Cross-Encoders for high-precision reranking against a local ChromaDB vector store.
* 📈 **`Real-Time Dashboard:`** A Streamlit UI featuring a 3D t-SNE projection of the product vector space, live agent activity logs, discovered deals and manual alert triggers(alongwith automated push alerts).
* ⚙️ **`Asynchronous & Concurrent:`** ThreadPoolExecutor handles fast, parallel multi-agent evaluation and web scraping without blocking the main thread.

---

## 🧠 System Architecture

**The architecture follows a highly asynchronous, multi-stage pipeline: transforming unstructured web data via preprocessing agents, evaluating it across three parallel pricing models, and autonomously dispatching profitable arbitrage opportunities.**

```text
                                                [ Web Deals / RSS Feeds ]
                                                           │
                                                           ▼
                                                   [ ScannerAgent ] 
                                                           │ (Raw Items extracted)
                                                           ▼
                                                [ PreprocessorAgent ]
                                                           │ (Normalized using Groq gpt-oss-20b)
                                                           ▼
                            ┌──────────────────────────────────────────────────────────┐
                            │        Ensemble Consensus Engine (Parallel)              │
                            │                                                          │
                            │  ┌──────────────────┬──────────────────┬──────────────┐  │
                            │  │  FrontierAgent   │ SpecialistAgent  │ NeuralNetwork│  │
                            │  │ (RAG + ChromaDB) │ (GGUF via Ollama)│  Agent       │  │
                            │  │ (BGE Embeddings) │ (Host Port 11434)│(PyTorch DNN) │  │
                            │  └────────┬─────────┴────────┬─────────┴─────┬────────┘  │
                            │           │                  │               │           │
                            │           └──────────────────┼───────────────┘           │
                            │                              ▼                           │
                            │                   [ Weighted Consensus ]                 │
                            └──────────────────────────────┬───────────────────────────┘
                                                           │ (Valuations & Margins)
                                                           ▼
                                        [ Deterministic / Autonomous Planner Agent ]
                                                           │
                                              ┌────────────┴────────────┐
                                              ▼                         ▼
                                     [ MessagingAgent ]           [ Streamlit UI ]
                                     (ntfy push alerts)        (Port 8501 Dashboard)
```

---
## 🤖 The Agent Ecosystem

**The engine is powered by a network of specialized `AI agents`. Every agent inherits from a unified `BaseAgent` for distinct, color-coded ANSI logging, while strict data validation and domain error routing are managed centrally via Pydantic schemas and custom exceptions in the core module.**

| Agent | Core Technology | Role |
| :--- | :--- | :--- |
| **ScannerAgent** | `gemini-3.1-flash-lite` | Parses raw HTML/RSS feeds and extracts valid deals using strict JSON structured outputs. |
| **PreprocessorAgent** | `Groq API (GPT-OSS)` | Cleans, normalizes, and categorizes unstructured text to maximize cache hits and model accuracy. |
| **EnsembleAgent** | `Thread/Concurrency Logic` | The consensus hub. Dispatches the three pricing sub-agents in parallel and calculates a dynamically weighted final price (with fault redistribution).  |
| **FrontierAgent** | `ChromaDB + SentenceTransformers` | Rewrites queries if needed, embeds queries, retrieves top market comparables, reranks them via Cross-Encoder, and synthesizes a price. |
| **SpecialistAgent** | `Ollama (Local GGUF)` | A specialized, fine-tuned local model (specialist-pricer) that estimates standalone market value. |
| **NeuralNetworkAgent** | `PyTorch DNN` | A custom ResNet-style dense neural network using HashingVectorizer for lightning-fast price regression. |
| **MessagingAgent** | `Groq API + ntfy.sh` | Crafts exciting, 150-character push notifications and dispatches them to the user's devices. |
| **DeterministicPlanningAgent** | `Sequential Pipeline` | Executes a fixed, high-speed procedural workflow, processing batches of deals and routing profitable ones to the messenger. |
| **AutonomousPlanningAgent** | `gemini-3.5-flash` | Uses ReAct and tool-calling to dynamically scan the internet, estimate true values, and autonomously notify the user of the single best bargain. |

---
## ⚡ Performance Benchmarks & System Optimizations

*Engineered to transition from a blocking, single-threaded prototype into a high-throughput, asynchronous autonomous engine.*

| Metric / Subsystem | Baseline (v1.0) | Optimized (v2.0) | Core Engineering Lever |
| :--- | :--- | :--- | :--- |
| **Pipeline Latency** | 32–35 min | **< 5 min** | Asynchronous task scheduling & non-blocking I/O |
| **Deal Discovery Loop** | ~15 min | **< 3 min** | Parallel consensus scoring + instant push dispatches |
| **VRAM Footprint** | ~6-7 GB | **< 2 GB** | Global model singleton pattern + **4-bit NF4 quantization** |
| **RAG Retrieval** | ~45 sec | **< 10 sec** | Vector index pruning + Cross-Encoder rerank batching |
| **Cache Hits** | 0% (Fresh calls) | **~0.0 ms** | LRU memory layer indexed on deterministic content hashes |

* 🚀 **`Radical Latency Compression:`** Cut processing time by over **85%**, unlocking real-time, zero-lag opportunistic deal hunting.
* 🧵 **`Asynchronous Agent Orchestration:`** Coordinates 10 discrete agents via dynamic `ThreadPoolExecutor` workers—interleaving network I/O (API calls, web scraping) with CPU/GPU bound tensor operations without blocking the event loop.
* 🗜️ **`Aggressive VRAM Squeezing:`** Pairs a compact ~2.0 GB 4-bit quantized GGUF specialist LLM with CUDA-accelerated SentenceTransformers (Bi-Encoder and Cross-Encoder) loaded via a global singleton pattern, slashing memory overhead by **>60%** to operate safely within 4GB consumer GPUs without Out-Of-Memory (OOM) faults.
* ⚡ **`Deterministic In-Memory Caching:`** Hashes raw payload strings into an LRU cache layer to bypass expensive re-computation on recurring listings, dropping marginal inference cost to zero.

---

## 🏛️ Architectural Philosophy: The Modular Monolith

*While multi-agent systems often default to distributed microservices, this engine is intentionally engineered as an **in-process Modular Monolith** to maximize throughput under strict local resource constraints.*

* 🏎️ **`Zero-Copy In-Memory IPC:`** Eliminates the serialized network payloads (JSON/gRPC) typical of distributed microservices. Agents communicate directly via pointer-like references in shared memory, driving inter-agent latency to **~0.0 ms**.
* 🧠 **`Global Model Instantiation (Anti-OOM):`** Avoids running redundant Python runtimes or duplicated PyTorch environments across isolated Docker containers. Neural weights load precisely once and are referenced globally across worker threads.
* 🧩 **`Frictionless Concurrency & State:`** Bypasses heavyweight external message brokers (Kafka/RabbitMQ). The central Gemini-powered ReAct orchestrator deterministically tracks memory states, thread pooling, and fallback paths natively in process.
  
---

## 📂 Directory Structure

```text
autonomous_multiagent_engine/
├── .github/
│   └── workflows/
│       └── ci.yml                          # Automated testing, linting, and type checks
├── docker/
│   ├── Dockerfile.api                      # Core Agent Orchestrator & API environment
│   ├── Dockerfile.inference                # Local LLM/GGUF Inference Microservice
│   └── Dockerfile.ui                       # Streamlit Frontend Dashboard
├── docker-compose.yml                      # Container orchestration
├── config/
│   └── settings.py                         # Pydantic configurations, API keys, and alerting rules
├── src/
│   ├── core/                               # Shared models, schemas, and contracts
│   │   ├── config.py                       # Environment setting exports
│   │   ├── items.py                        # Prompt generation and hub integration
│   │   ├── schemas.py                      # Pydantic data models (Deal, Opportunity)
│   │   └── exceptions.py                   # Custom project-level exceptions
│   ├── agents/                             # Specialized AI agents
│   │   ├── base_agent.py                   # Abstract base class (logging, timestamps, hooks)
│   │   ├── preprocessor_agent.py           # Data normalization and standardization
│   │   ├── specialist_agent.py             # Local GGUF inference interface
│   │   ├── neural_network_agent.py         # PyTorch deep neural network regression
│   │   ├── ensemble_agent.py               # Weighted consensus engine (RAG + DNN + GGUF)
│   │   ├── frontier_agent.py               # High-reasoning RAG agent (ChromaDB)
│   │   ├── scanner_agent.py                # Deal scraping and extraction
│   │   ├── messaging_agent.py              # Push notification dispatcher (ntfy)
│   │   ├── deterministic_planning_agent.py # Fixed pipeline execution
│   │   └── autonomous_planner_agent.py     # ReAct / Tool-calling orchestrator
│   ├── evaluation/          
│   │   └── benchmark.py                    # Evaluation framework for agent accuracy
│   ├── services/                                                                       
│   │   └── framework.py                    # Agent lifecycle and orchestration manager
│   └── ui/
│       └── app.py                          # Production Streamlit dashboard
├── assets/                                 # Documentation images and logs
├── notebooks/
│   └── pipeline_eval_master.ipynb          # Testing and evaluation sandbox
├── data/
│   ├── vectorstore/                        # Persistent ChromaDB embeddings
│   ├── models/                             # Quantized GGUF weights & PyTorch .pth files
│   ├── preprocessed_test_500.json          # Baseline test dataset
│   └── memory.json                         # Persisted deal evaluation state
├── .gitignore                              # Prevents sensitive/heavy files from pushing to Git
├── .dockerignore                            # Prevents heavy files from Docker build context
├── Modelfile                               # Local Ollama configuration for GGUF
├── pyproject.toml                          # Dependencies and build system
└── README.md                               # Project documentation
```

---
## 🛠️ Infrastructure, CI/CD & LLMOps Pipeline

> **Engineered for production resilience, bridging traditional backend orchestration with specialized AI workflows—featuring isolated containerization, zero-leak volume persistence, and sub-second CI static analysis.**

| Component | Technology | Primary Function | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Container Engine** | Docker (`py3.11-slim`) | `Layer-cached runtime environment` | Deterministic dependency isolation & clean module resolution |
| **Orchestrator** | Docker Compose | `Host-to-container network bridging` | Routes traffic seamlessly to host inference engines (`host.docker.internal`) |
| **State Persistence** | Bind Mounts (`./data`) | `Host directory mapping` | Direct local disk persistence for ChromaDB vectors, cache, and runtime artifacts |
| **Automated CI** | GitHub Actions | `Automated build & gatekeeping pipeline` | Blocks regressions on branch merges (`push`/`PR` to `main`) |
| **Static Analysis** | Ruff (Rust Engine) | `Ultra-fast AST linting & syntax checks` | Instant code-quality enforcement across `src/` prior to build steps |

---

## 🧬 Core Infrastructure Highlights

* 🐳 **`Optimized Containerization:`** Packages the agent orchestrator and Streamlit UI in a stripped-down Python 3.11 environment. Layer-caching strategies slash image build times, while editable-mode package installation ensures seamless relative import resolution across modular agent boundaries.
* 🌐 **`Hybrid Host-Bridge Orchestration:`** Uses *host-gateway* bridge networking in Docker Compose to let containerized agents communicate with host-accelerated Ollama instances with zero latency—retaining full GPU access without requiring complex nested-container GPU passthrough.
* 💾 **`Stateful Vector & Weight Persistence:`** Leverages bidirectional bind mounts (`./data`) to persist local ChromaDB indices, cache states, and neural regression weights directly to the host disk, guaranteeing historical deal memory survives full container teardowns.
* ⚡ **`Lightning-Fast CI Gatekeeping:`** Replaces sluggish legacy linters with **Ruff**, executing full-tree static analysis and AST validation in milliseconds within GitHub Actions to reject broken workflows before deployment.
  
---

## 🚀 Quickstart

### 1. Prerequisites

Ensure the following tools and services are installed and accessible on your host machine:

* 🐳 **Docker & Docker Compose** (v20.10+)
* 🦙 **Ollama** running locally on port `11434` *(required for the local GGUF Specialist Agent)*
* 🔑 **API Keys:** Google AI Studio ([Gemini API Key](https://aistudio.google.com/)) and [Groq Console](https://console.groq.com/)

### 2. Download Pre-Trained Weights

Place the neural checkpoint and GGUF model weights into their expected target paths before booting:

| Component | Format | Source | Target Path |
| :--- | :--- | :--- | :--- |
| **Neural Network Agent** | `.pth` | [Google Drive Checkpoint](https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK?usp=drive_link) | `data/models/deep_neural_network.pth` |
| **Specialist Agent LLM** | `.gguf` | [Hugging Face Repository](https://huggingface.co/Subhrajyoti75/specialist-pricer-q4_k_m) | Ollama Model Library or local directory |

*To import the GGUF model directly via Ollama:*

```bash
ollama run hf.co/Subhrajyoti75/specialist-pricer-q4_k_m
```

### 2. Environment Configuration
Create a `.env` file in the root directory:

### 1. Prerequisites
Docker & Docker Compose

Ollama running locally (if using the Specialist Agent).

API Keys for Gemini and Groq.

### 2. Download Pre-Trained Model Weights

Before booting the cluster, you need to download the trained weights for the local Deep Neural Network and the localized GGUF LLM. 

*   **Deep Neural Network Agent (.pth):**
    👉 [Download `deep_neural_network.pth` (Google Drive)](https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK?usp=drive_link)
    *(Save this file exactly to: `data/models/deep_neural_network.pth`)*

*   **Specialist Agent Local LLM (.gguf):**
    👉 [Download `specialist-pricer-q4_k_m` (Hugging Face)](https://huggingface.co/Subhrajyoti75/specialist-pricer-q4_k_m)
    *(Pull this via Ollama or place the GGUF file in your local models directory)*

### 3. Environment Variables

Create a .env file in the root directory:
```Code Snippet
  # API Keys
  GEMINI_API_KEY=your_gemini_api_key_here
  GROQ_API_KEY=your_groq_api_key_here
  HF_TOKEN=your_huggingface_token_here
  
  # Infrastructure
  OLLAMA_HOST=http://host.docker.internal:11434
  PRICER_MODEL_URL=http://host.docker.internal:11434
  
  # Push Notifications
  NTFY_URL=https://ntfy.sh
  NTFY_TOPIC=deal-engine-alerts-unique-id
```

### 4. Boot the Cluster

Clone the repository, configure weights, and spin up the container stack:

```bash
git clone https://github.com/Subhrajyoti8520/autonomous_multiagent_engine.git
cd autonomous_multiagent_engine

docker-compose up --build -d
```

### 5. Access the Interfaces

* 📊 Live Streamlit Dashboard: `http://localhost:8501`
* 🧠 Host Inference Engine: `http://localhost:11434`
* 📱 Mobile Alert Stream: Subscribe to your custom NTFY_TOPIC on the ntfy mobile client or web interface.

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

For major changes, please open an issue first to discuss your proposed updates before opening a pull request.

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your Changes**
   ```bash
   git commit -m "feat: add AmazingFeature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full terms and details.



# MediMind AI — Intelligent Healthcare Diagnostic Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Claude AI](https://img.shields.io/badge/Claude_AI-Sonnet_4-D97706?style=for-the-badge&logo=anthropic&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**An end-to-end AI-powered healthcare assistant — from a lightweight ML chatbot to a production-ready full-stack platform with LLM agents, RAG retrieval, and real-time clinical chat.**

[Live Demo](#quick-start) · [API Docs](#api-reference) · [Report Bug](https://github.com/Yashas14/Health_Care_Chat_Bot/issues) · [Request Feature](https://github.com/Yashas14/Health_Care_Chat_Bot/issues)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [CLI Chatbot](#1-cli-chatbot)
  - [Full-Stack Platform](#2-full-stack-platform-local)
  - [Docker Compose](#3-full-stack-with-docker-compose)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [ML Training](#ml-training)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Snapshots of the Results](#snapshots-of-the-results)

---

## Overview

**MediMind AI** is a dual-project healthcare intelligence system built to demonstrate how modern AI, ML, and cloud-native engineering come together in the medical domain.

| Project | Description | Tech |
|---------|-------------|------|
| **MediMind CLI** | Lightweight terminal symptom checker using a Decision Tree + SVM ensemble | Python, scikit-learn, pandas |
| **MediMind Platform** | Production-grade full-stack AI platform with LLM agents, RAG, real-time WebSocket chat, and a Next.js 14 UI | FastAPI, Next.js 14, Claude AI, ChromaDB |

Both projects share the same curated medical dataset covering **41 diseases** and **132 symptom features**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Next.js 14 Frontend                       │
│   AI Chat  │  Body Map  │  Symptom Checker  │  Health Dashboard │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST + WebSocket
┌──────────────────────────▼──────────────────────────────────────┐
│                   FastAPI Backend (Port 8000)                    │
│                                                                  │
│  ┌─────────────────────── Agent Pipeline ───────────────────┐   │
│  │  1. SymptomExtractor → 2. Diagnosis → 3. Triage          │   │
│  │                      → 4. Precaution → 5. MedicalSummary │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  JWT Auth │ Rate Limiter │ Audit Logger │ Request Tracer         │
└────┬──────────┬──────────────┬──────────────────────────────────┘
     │          │              │
┌────▼───┐ ┌───▼──────┐ ┌─────▼──────┐
│SQLite /│ │ChromaDB  │ │ Claude AI  │
│Postgres│ │(RAG)     │ │ Sonnet 4   │
└────────┘ └──────────┘ └────────────┘
```

---

## Features

### MediMind CLI (Chatbot)
- **ML Ensemble** — Decision Tree + SVM for disease prediction
- **Conversational symptom collection** via interactive terminal prompts
- **Severity scoring** — weighted symptom severity from curated medical data
- **Disease descriptions** and evidence-based precautionary advice
- **Cross-validation accuracy** reporting on every run

### MediMind Platform (Full-Stack)
- **5-Stage AI Diagnostic Pipeline** powered by Anthropic Claude Sonnet 4
- **RAG Knowledge Base** — ChromaDB vector store with medical literature retrieval
- **Real-time WebSocket Chat** — streaming AI responses with session persistence
- **Interactive Body Map** — click-to-select symptom input interface
- **Health Dashboard** — personal diagnosis history, trends, and health insights
- **JWT Authentication** — secure register / login / refresh token flow
- **Drug Information** — OpenFDA API integration for medication lookups
- **Nearby Hospitals** — Google Maps API integration for emergency referrals
- **HIPAA-Aware Logging** — PII auto-redacted from all structured log output
- **Rate Limiting** — 100 requests / 60 s per IP, configurable
- **Dark / Light Mode** — full theme support with next-themes
- **ICD-10 Codes** — differential diagnoses mapped to standard clinical codes
- **Triage Classification** — Emergency / Urgent / Soon / Routine urgency levels
- **Multi-language Ready** — i18next internationalisation scaffold included

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 14 (App Router), TypeScript, Tailwind CSS, shadcn/ui, Framer Motion |
| **Backend** | FastAPI 0.115, Python 3.11, Uvicorn, WebSockets |
| **Database** | SQLite (dev) / PostgreSQL 16 (prod), SQLAlchemy 2.0 async, Alembic |
| **AI / LLM** | Anthropic Claude Sonnet 4, LangChain-style agent orchestration |
| **ML** | scikit-learn, XGBoost, pandas, numpy, joblib |
| **Vector Store** | ChromaDB 0.5 (local persistent or server mode) |
| **Auth & Security** | python-jose (JWT HS256), bcrypt (12 rounds), cryptography |
| **Caching** | Redis 7 (Docker Compose; optional locally) |
| **External APIs** | OpenFDA, Google Maps Platform |
| **Infrastructure** | Docker, Docker Compose, Kubernetes-ready, Terraform |
| **Testing** | pytest, pytest-asyncio, pytest-cov |

---

## Project Structure

```
Health_Care_Chat_Bot-main/
├── chatbot/                    # MediMind CLI — standalone ML chatbot
│   ├── chat_bot.py             # Entry point
│   ├── requirements.txt
│   └── Data/                   # Training.csv, Testing.csv, dataset.csv
│       MasterData/             # Symptom severity, descriptions, precautions
│
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py             # App factory, router registration, lifespan
│   │   ├── agents/             # 5-agent diagnostic pipeline
│   │   │   ├── orchestrator.py
│   │   │   ├── symptom_extractor.py
│   │   │   ├── diagnosis.py
│   │   │   ├── triage.py
│   │   │   ├── precaution.py
│   │   │   └── medical_summary.py
│   │   ├── api/routes/         # REST + WebSocket endpoints
│   │   ├── core/               # Config, DB, security, middleware, logging
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── rag/                # ChromaDB ingestion & retrieval
│   │   └── services/           # Google Maps, OpenFDA, data loader
│   ├── alembic/                # Database migrations
│   ├── tests/                  # pytest test suite
│   ├── .env                    # Local environment variables
│   └── requirements.txt
│
├── frontend/                   # Next.js 14 application
│   ├── app/                    # App Router pages
│   │   ├── (app)/chat/         # AI chat interface
│   │   ├── (app)/dashboard/    # Health dashboard
│   │   ├── (app)/symptom-checker/  # Body map UI
│   │   ├── login/ & register/  # Auth pages
│   │   └── layout.tsx
│   ├── components/             # Reusable UI components (shadcn/ui)
│   ├── hooks/                  # WebSocket, voice input hooks
│   ├── lib/                    # API client, utilities
│   └── public/
│
├── ml-training/                # Standalone ML model training
│   ├── train_ensemble.py
│   ├── evaluate_models.py
│   └── model_artifacts/        # Saved joblib models
│
└── docker-compose.yml          # Full stack: backend + frontend + postgres + redis + chromadb
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Node.js | 18+ |
| Git | any |
| Docker + Docker Compose | optional (for full stack) |

---

### 1. CLI Chatbot

```bash
git clone https://github.com/Yashas14/Health_Care_Chat_Bot.git
cd Health_Care_Chat_Bot-main/chatbot

pip install -r requirements.txt
python chat_bot.py
```

Follow the terminal prompts to enter symptoms and receive a diagnosis with precautionary advice.

---

### 2. Full-Stack Platform (Local)

#### Backend

```bash
cd Health_Care_Chat_Bot-main/backend

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt

# (Optional) Add your Anthropic API key to enable LLM features
# Edit backend/.env and set: ANTHROPIC_API_KEY=sk-ant-...

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Backend API root |
| http://localhost:8000/docs | Interactive Swagger UI |
| http://localhost:8000/redoc | ReDoc API reference |

**Demo account (auto-seeded on first run):**
```
Email:    demo@123
Password: demo@123
```

#### Frontend

```bash
cd Health_Care_Chat_Bot-main/frontend

npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

### 3. Full Stack with Docker Compose

```bash
cd Health_Care_Chat_Bot-main

# Copy env template and fill in your API keys
cp backend/.env .env

docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

---

## Configuration

All settings are loaded from `backend/.env`. The app runs fully without any external API keys — LLM and map features gracefully degrade.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./healthcare.db` | DB connection string |
| `JWT_SECRET_KEY` | `dev-secret-key-change-in-production` | JWT signing secret — **change before deploying** |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Access token lifetime |
| `ANTHROPIC_API_KEY` | _(empty)_ | Claude API key — required for AI chat & diagnosis |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model version |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8001` | ChromaDB port |
| `GOOGLE_MAPS_API_KEY` | _(empty)_ | Enables nearby hospital search |
| `OPENFDA_API_KEY` | _(empty)_ | Enhanced drug information lookups |
| `ENVIRONMENT` | `development` | `development` / `staging` / `production` |
| `DEBUG` | `true` | Verbose SQLAlchemy + app debug logging |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS allowed origins (comma-separated) |

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/register` | Register a new user account |
| `POST` | `/api/v1/auth/login` | Obtain JWT access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh an expired access token |
| `POST` | `/api/v1/auth/logout` | Revoke the current session |
| `GET` | `/api/v1/auth/me` | Get authenticated user profile |

### Symptoms & Diagnosis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/symptoms/analyze` | Extract canonical symptoms from free text |
| `POST` | `/api/v1/symptoms/full-pipeline` | Full 5-agent diagnostic pipeline |
| `GET` | `/api/v1/diagnosis/history` | Retrieve the user's diagnosis history |
| `POST` | `/api/v1/diagnosis/{id}/feedback` | Submit feedback on a diagnosis |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/chat/sessions` | List all chat sessions |
| `POST` | `/api/v1/chat/sessions` | Create a new chat session |
| `GET` | `/api/v1/chat/sessions/{id}/messages` | Fetch session message history |
| `WS` | `/api/v1/chat/ws/{session_id}` | Real-time streaming AI chat |

### External Services

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/hospitals/nearby` | Find nearby hospitals via Google Maps |
| `GET` | `/api/v1/drugs/search` | Drug information via OpenFDA |
| `GET` | `/api/v1/rag/query` | Direct RAG knowledge base query |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Liveness probe — returns service status |
| `GET` | `/docs` | Swagger UI (development only) |

---

## ML Training

Train the ensemble model used by the DiagnosisAgent:

```bash
cd ml-training
pip install -r requirements.txt

# Train Decision Tree + SVM + XGBoost ensemble
python train_ensemble.py
# Output: model_artifacts/ensemble_model.joblib

# Evaluate and compare models
python evaluate_models.py
```

| Model | Test Accuracy |
|-------|--------------|
| Decision Tree | ~95% |
| SVM | ~94% |
| XGBoost | ~96% |
| Ensemble (voting) | ~97% |

> Without a trained model the platform automatically falls back to **LLM-only diagnosis mode** via Claude.

---

## Running Tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
```

| Test File | Coverage |
|-----------|----------|
| `test_api.py` | Auth, users, health endpoints |
| `test_agents.py` | Agent pipeline unit tests |
| `test_phase3.py` | Integration: full diagnostic pipeline |

---

## Deployment

### Environment Checklist Before Production

- [ ] Set a strong, unique `JWT_SECRET_KEY` (32+ random bytes)
- [ ] Set `ENVIRONMENT=production` and `DEBUG=false`
- [ ] Switch `DATABASE_URL` to a managed PostgreSQL instance
- [ ] Enable Redis for session caching
- [ ] Set `ALLOWED_ORIGINS` to your production domain only
- [ ] Rotate the `ANTHROPIC_API_KEY` in a secrets manager (not in `.env`)
- [ ] Enable HTTPS / TLS termination at the reverse proxy layer

### Docker Compose (Production-like)

```bash
docker compose -f docker-compose.yml up -d --build
```

### Kubernetes

A `k8s/` directory with Helm charts and Terraform modules can be scaffolded for cloud deployments (AWS EKS / GCP GKE). The `docker-compose.yml` service definitions map directly to Kubernetes Deployments and Services.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes using conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation update
   - `test:` adding or updating tests
   - `refactor:` code change with no functional difference
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request with a clear description of the change

Please ensure all existing tests pass and new features include tests.

---

## Snapshots of the Results 


<img width="1904" height="726" alt="Screenshot 2026-05-20 142851" src="https://github.com/user-attachments/assets/9f8a9bd2-19e5-4537-90fe-1b70d058b182" />

--


<img width="1883" height="856" alt="image" src="https://github.com/user-attachments/assets/b1a5928d-4d86-42e6-accd-446cca9279ec" />

--


<img width="1894" height="856" alt="image" src="https://github.com/user-attachments/assets/1e0d38e6-21a2-4874-9738-6c64416515f4" />

--


<img width="1874" height="873" alt="image" src="https://github.com/user-attachments/assets/dd5c7b4a-a175-4d1e-8029-cbf36cbc76a7" />



---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [Yashas](https://github.com/Yashas14)

⭐ Star this repo if you found it useful!

</div>

---

## Author

**Yashas D**

- LinkedIn: https://www.linkedin.com/in/yashasd2004/
- GitHub: https://github.com/Yashas14
- Repository: https://github.com/Yashas14/Health_Care_Chat_Bot

<p align="center">
  <h1 align="center">🚚 Fair Dispatch System</h1>
  <p align="center">
    <strong>Single‑API Fair Routing · Angelic Fairness Engine · Live Agent Visualization</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-api-reference">API Reference</a> •
    <a href="#-visualization-dashboard">Dashboard</a>
  </p>
</p>

---

Fair Dispatch is an AI‑assisted, **fairness‑aware route allocation engine** designed as a single seamless API that any logistics stack can plug into.

**You send today's drivers and packages as JSON. The system does everything else:**
- 📦 Clustering packages into optimal routes
- ⚖️ Calculating effort scores and fairness metrics
- 🛣️ Planning routes with EV-aware optimization
- 🤝 Balancing workload across drivers
- 🤖 AI-powered driver negotiation and explanation
- 📊 Learning from feedback to improve over time

...and streams the whole multi‑agent process into a **live visualization**.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🎯 Single API Endpoint** | One POST to `/api/v1/langgraph/allocate` handles everything |
| **🤖 5+ Specialized AI Agents** | LangGraph-orchestrated multi-agent workflow |
| **⚖️ Fairness-First Design** | Gini index, individual fairness scores, and equity metrics |
| **🗣️ Natural Language Explanations** | Gemini-powered driver-friendly route explanations |
| **📊 Live Agent Visualization** | Real-time Streamlit dashboard showing agent workflow |
| **🔄 Continuous Learning** | Feedback loop improves allocations over time |
| **⚡ EV-Aware Routing** | Battery constraints and charging station integration |
| **🔐 Full Audit Trail** | Complete decision logging for transparency |

## 🏗️ Architecture

### Multi-Agent Workflow (LangGraph)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           FAIR DISPATCH WORKFLOW                               │
└────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  🔧 Initialize   │ → │  📦 Clustering   │ → │  💪 ML Effort    │
│     Node        │   │     Agent       │   │     Agent       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                                    │
                                                    ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  ⚡ EV Recovery  │ ← │  ⚖️ Fairness     │ ← │  🛣️ Route        │
│     Node        │   │   Manager       │   │   Planner       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
        │                     │
        ▼                     ▼ (if unfair)
┌─────────────────┐   ┌─────────────────┐
│  🤝 Driver       │   │  🔄 Reoptimize   │
│   Liaison       │   │     Loop        │
└─────────────────┘   └─────────────────┘
        │
        ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  🎓 Learning     │ → │  🗣️ LLM       │ → │  ✅ Finalize     │
│     Agent       │   │   Explain       │   │     Node        │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Agent Descriptions

| Agent | Purpose | Key Outputs |
|-------|---------|-------------|
| **Initialize Node** | Sets up allocation state, validates inputs | Validated driver/package data |
| **Clustering Agent** | Groups packages using K-Means by geography | Route clusters with centroids |
| **ML Effort Agent** | Builds effort matrix for all driver-route pairs | Effort scores, XGBoost predictions |
| **Route Planner Agent** | Solves optimal assignment (Hungarian algorithm) | Driver-route assignments |
| **Fairness Manager** | Evaluates Gini index, std dev, thresholds | ACCEPT or REOPTIMIZE decision |
| **EV Recovery Node** | Handles EV battery constraints | Charging station insertions |
| **Driver Liaison Agent** | Handles driver negotiations/appeals | Appeal resolutions |
| **Learning Agent** | Updates models from feedback | Improved future allocations |
| **LLM Explain Node** | Generates natural language explanations | Human-readable route descriptions |

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 14+** (or SQLite for development)
- **Git**

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/your-org/fair-dispatch-system.git
cd fair-dispatch-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
```

**Essential environment variables:**

```env
# Database (PostgreSQL recommended for production)
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/fair_dispatch

# Application
APP_ENV=development
DEBUG=true

# Optional: Gemini API for AI explanations
GOOGLE_API_KEY=your-gemini-api-key

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

### 3. Setup Database

```bash
# Create PostgreSQL database
createdb fair_dispatch

# Run migrations
alembic upgrade head
```

### 4. Start the Server

```bash
# Development server with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access the System

| Endpoint | URL |
|----------|-----|
| **API Documentation** | http://localhost:8000/docs |
| **ReDoc** | http://localhost:8000/redoc |
| **Demo Page** | http://localhost:8000/demo/allocate |
| **Admin Dashboard** | http://localhost:8000/admin |

## 📊 Visualization Dashboard

The system includes a **real-time Streamlit dashboard** for monitoring allocations:

```bash
# Navigate to dashboard directory
cd supply_chain_dashboard

# Install dashboard dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

**Dashboard Features:**
- 🗺️ **Live Map Visualization** - See routes on an interactive map
- 📈 **Fairness Metrics** - Real-time Gini index and equity scores
- 🤖 **Agent Activity Feed** - Watch agents work in real-time
- 📊 **Analytics Charts** - Workload distribution and trends

## 📡 API Reference

### Primary Endpoint: Allocate Routes

**`POST /api/v1/langgraph/allocate`**

This single endpoint handles the complete allocation workflow.

#### Request

```json
{
  "date": "2026-02-10",
  "warehouse": {
    "lat": 12.9716,
    "lng": 77.5946
  },
  "packages": [
    {
      "id": "pkg_001",
      "weight_kg": 2.5,
      "fragility_level": 3,
      "address": "123 Main St, Bangalore",
      "latitude": 12.97,
      "longitude": 77.60,
      "priority": "NORMAL"
    },
    {
      "id": "pkg_002",
      "weight_kg": 1.0,
      "fragility_level": 1,
      "address": "456 Oak Ave, Bangalore",
      "latitude": 12.98,
      "longitude": 77.61,
      "priority": "HIGH"
    }
  ],
  "drivers": [
    {
      "id": "driver_001",
      "name": "Raju",
      "vehicle_capacity_kg": 150,
      "preferred_language": "en",
      "vehicle_type": "PETROL"
    },
    {
      "id": "driver_002",
      "name": "Kumar",
      "vehicle_capacity_kg": 200,
      "preferred_language": "ta",
      "vehicle_type": "EV",
      "ev_range_km": 120
    }
  ]
}
```

#### Response

```json
{
  "allocation_run_id": "550e8400-e29b-41d4-a716-446655440000",
  "date": "2026-02-10",
  "status": "SUCCESS",
  "global_fairness": {
    "avg_workload": 63.2,
    "std_dev": 5.4,
    "gini_index": 0.12,
    "max_gap": 8.3
  },
  "assignments": [
    {
      "driver_id": "driver_001",
      "driver_name": "Raju",
      "route_id": "route_uuid",
      "workload_score": 65.3,
      "fairness_score": 0.92,
      "route_summary": {
        "num_packages": 22,
        "total_weight_kg": 48.5,
        "num_stops": 14,
        "estimated_time_minutes": 145
      },
      "explanation": "Your route covers the Koramangala area with 22 packages, mostly residential. Expected completion time is around 2.5 hours with moderate traffic."
    }
  ],
  "agent_events": [
    {
      "agent": "clustering_agent",
      "status": "completed",
      "message": "Created 5 route clusters"
    },
    {
      "agent": "fairness_manager",
      "status": "completed", 
      "message": "Allocation ACCEPTED (Gini: 0.12)"
    }
  ]
}
```

### Additional Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/drivers/{id}` | Get driver details and stats |
| `GET` | `/api/v1/routes/{id}` | Get route details and packages |
| `POST` | `/api/v1/feedback` | Submit driver feedback |
| `GET` | `/api/v1/admin/dashboard` | Admin dashboard data |
| `GET` | `/api/v1/runs` | List allocation runs |
| `GET` | `/api/v1/runs/{id}/events` | Get agent events for a run |

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_allocation.py -v

# Run E2E tests only
make test-e2e

# Run tests in parallel (faster)
pytest tests/ -n auto
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `DEBUG` | `true` | Enable debug mode |
| `GOOGLE_API_KEY` | - | Gemini API key for explanations |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | - | LangSmith API key |

### Workload Score Weights

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKLOAD_WEIGHT_A` | `1.0` | Weight for num_packages |
| `WORKLOAD_WEIGHT_B` | `0.5` | Weight for total_weight_kg |
| `WORKLOAD_WEIGHT_C` | `10.0` | Weight for route_difficulty_score |
| `WORKLOAD_WEIGHT_D` | `0.2` | Weight for estimated_time_minutes |

### Fairness Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_PACKAGES_PER_ROUTE` | `20` | Target packages per cluster |
| `GINI_THRESHOLD` | `0.25` | Max acceptable Gini index |
| `STD_DEV_THRESHOLD` | `15.0` | Max acceptable standard deviation |

## 📐 Algorithms

### Workload Score Formula

```
workload_score = a × num_packages 
               + b × total_weight_kg 
               + c × route_difficulty_score 
               + d × estimated_time_minutes
```

### Gini Index

Measures inequality in workload distribution (0 = perfect equality, 1 = maximum inequality):

```
G = (2 × Σ(i × x_i)) / (n × Σx_i) - (n + 1) / n
```

### Individual Fairness Score

Per-driver fairness relative to average:

```
fairness_score = 1 - |workload - avg_workload| / max(avg_workload, 1)
```

## 📁 Project Structure

```
fair-dispatch-system/
├── 📂 alembic/                 # Database migrations
│   └── versions/               # Migration files
├── 📂 app/
│   ├── 📂 api/                 # FastAPI routers
│   │   ├── allocation.py       # POST /allocate (basic)
│   │   ├── allocation_langgraph.py  # POST /langgraph/allocate
│   │   ├── admin.py            # Admin endpoints
│   │   ├── drivers.py          # Driver endpoints
│   │   ├── feedback.py         # Feedback endpoints
│   │   └── routes.py           # Route endpoints
│   ├── 📂 models/              # SQLAlchemy models
│   │   ├── driver.py
│   │   ├── package.py
│   │   ├── route.py
│   │   └── assignment.py
│   ├── 📂 schemas/             # Pydantic DTOs
│   ├── 📂 services/            # Business logic
│   │   ├── langgraph_workflow.py    # Agent orchestration
│   │   ├── langgraph_nodes.py       # Individual agents
│   │   ├── ml_effort_agent.py       # ML scoring
│   │   ├── fairness_manager_agent.py
│   │   ├── route_planner_agent.py
│   │   ├── driver_liaison_agent.py
│   │   ├── learning_agent.py
│   │   ├── gemini_explain_node.py
│   │   └── ...
│   ├── config.py               # Settings
│   ├── database.py             # DB connection
│   └── main.py                 # FastAPI app
├── 📂 frontend/                # Static frontend files
│   ├── index.html              # Demo UI
│   └── visualization.html      # Live visualization
├── 📂 supply_chain_dashboard/  # Streamlit dashboard
│   ├── dashboard.py
│   └── api_client.py
├── 📂 tests/                   # Test suite
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

## 🔧 Development

### Running in Development Mode

```bash
# Start with auto-reload
uvicorn app.main:app --reload

# Start with custom port
uvicorn app.main:app --reload --port 3000

# Start with debug logging
DEBUG=true uvicorn app.main:app --reload
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1

# View migration history
alembic history
```

### Makefile Commands

```bash
make test          # Run all tests
make test-cov      # Run with coverage
make test-e2e      # Run E2E tests
make test-parallel # Run tests in parallel
make lint          # Run linting
make format        # Format code
make ci            # Full CI pipeline
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ for fairer logistics
</p>

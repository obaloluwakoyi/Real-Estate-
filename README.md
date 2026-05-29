# ⚛️ Quantum Portfolio Intelligence Terminal

A production-grade, institutional risk management and strategic asset allocation workspace. This application leverages hybrid quantum-inspired optimization topologies via **Qiskit**, adaptive stochastic forward volatility forecasting modules (**Geometric Brownian Motion + Merton Jump-Diffusion**), and automated cognitive AI risk reporting engines.

🔗 **Live Production Demo**: [Access Terminal Workspace](https://multi-asset-quantum-ai-portfolio-terminal-b9qp8svq4s6sh2pasax7.streamlit.app/)

---

## 🏛️ System Architecture & Pipeline Framework

The core architecture operates as a strict serial pipeline executing four decoupled quantitative modules:

1. **Telemetry Feed Engine**: Ingests real-time financial data matrices via Yahoo Finance API, purging invalid entries and computing drift ($\mu$) and variance-covariance ($\Sigma$) parameters.
2. **Quantum/Classical Layer**: Utilizes a Variational Quantum Eigensolver (**VQE** via Qiskit) optimized via `COBYLA` to map portfolio allocations to a Quadratic Program matrix, with an elegant algorithmic fallback structure.
3. **Stochastic Predictive Forecasting**: Simulates forward multi-year trajectory pathways over a 5-year window, accounting for regime switches or macro shocks using Poisson jump-diffusion profiles.
4. **Risk Guard & Cognitive Agent**: Executes multi-tier compliance stress testing (95% VaR / 95% CVaR), maps localized volatility exit conditions, and pipes structured payloads to an LLM provider for cold, institutional risk analysis reporting.

---

## 🛠️ File Registry & Component Breakdown

* **`app.py`**: The central UI workspace engineered in Streamlit. Contains layout styles, Plotly charting infrastructure for forward volatility pathways, and asynchronous caching configurations.
* **`pipeline.py`**: The foundational orchestrator. Handles cross-module data integration, error routing boundary definitions, and compiles asset telemetry charts into a cleanly typeset PDF export.
* **`quantum_engine.py`**: Hosts the mathematical optimization framework. Builds the `PortfolioOptimization` quadratic program and solves for optimum weights utilizing the `EfficientSU2` ansatz library.
* **`predict.py`**: Contains the stochastic simulation mathematical layers. Computes standard continuous Geometric Brownian Motion (GBM) as well as Merton Jump-Diffusion paths matching strict martingale constraints.
* **`compliance_engine.py`**: Performs deep portfolio forensics. Runs historical back-shocks (e.g., 2008 Systemic Meltdown) and maps localized asset trailing stop-losses dynamically.
* **`ai_analyst.py`**: The cognitive wrapper. Houses payload normalization rules and token routing interfaces for multiple API providers (Groq, Gemini, OpenRouter, Ollama) alongside an offline rule-based report generator.

---

## 🚀 Quickstart: Local Deployment Setup

Ensure you have Python 3.10+ installed on your local workstation.

### 1. Clone the Repository and Navigate to Root
```bash
git clone <(https://github.com/obaloluwakoyi/Multi-Asset-Quantum-AI-Portfolio-Termina)>
cd multi-asset-quantum-portfolio

2. Configure Your Virtual EnvironmentBashpython -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install Pinpoint DependenciesBashpip install -r requirements.txt

4. Boot Up the DashboardBashstreamlit run app.py

⚙️ Core Predictive Parameterizations
The terminal provides deep configuration dials over its predictive path frameworks via the control panel drawer:
Forecasting ModeMathematical Formulation ProfileIdeal Use Case ConditionsPure GBMContinuous log-normal paths utilizing historical drift ($\mu$) and standard deviation ($\sigma$).
Low-volatility, steady bull-market environments.Pure MertonIntroduces a Poisson process counter ($\lambda_j$) tracking sudden structural downside jumps ($\mu_j$).
Evaluating tail risk vulnerabilities.
Dynamic BlendLinearly blends path weights between continuous diffusion and sudden jump regimes.Balanced macroeconomic forecasting.
Trigger-Switch (VIX)Automatically flips from GBM to Merton Jump-Diffusion if simulated VIX $\ge 25.0$.High-stress environment stress-testing.


🔒 Environment Variable & Token InfrastructureThe terminal supports seamless runtime integration with API frameworks.
 If you prefer to skip API key configuration setups locally, select Offline / No-API inside the provider dropdown to bypass remote endpoint pipelines and routing logic.
To use LLM capabilities, input your secure keys inside the UI token fields for:Groq API: High-speed Llama3/Mixtral text inference.Gemini API: Deep context reasoning models.
Ollama Endpoint: Local offline models running on your local daemon infrastructure (http://localhost:11434).



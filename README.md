# ⚛️ Multi-Asset Quantum-AI Portfolio Terminal v3.0

An institutional-grade, dark-themed quantitative financial workspace built using a modern decoupled 5-file architecture. The system combines classical data telemetry, a hybrid quantum state variational optimizer (**Qiskit SamplingVQE**), advanced stochastic modeling (**Geometric Brownian Motion** and **Merton's Jump-Diffusion Model**), an institutional risk/compliance scanner, and cross-provider LLM cognitive architectures to generate actionable rebalancing strategy narratives and downloadable PDF executive reports.

---LIVE DEMO :- [https://qutum1.streamlit.app/](https://multi-asset-quantum-ai-portfolio-terminal-b9qp8svq4s6sh2pasax7.streamlit.app/)


## 🏗️ System Architecture & File Layout

The application decouples interface presentation elements cleanly from mathematical, stochastic, and structural code components using five dedicated processing nodes:

```text
QUANTUM_TERMINAL_APP/
├── app.py                # UI Layout Engine, Plotly visualizations, custom CSS styles, and reactive state nodes.
├── pipeline.py           # Core Workflow Orchestrator connecting metrics, trigger routing, and PDF compilation.
├── quantum_engine.py     # Classical data telemetry interfaces, hybrid Qiskit solvers, and continuous path generators.
├── predict.py            # Advanced stochastic prediction models (GBM, Merton Jump-Diffusion, and VIX switching).
├── compliance_engine.py  # Regulatory constraint scanners, Value-at-Risk (VaR/CVaR), and dynamic exit matrices.
└── ai_analyst.py         # Multi-provider LLM API router (Groq, Gemini, OpenRouter, Ollama) and offline rule fallback.

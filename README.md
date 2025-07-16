# Particle Accelerator Simulation

AI-powered particle physics laboratory that simulates LHC-like collisions and uses machine learning to discover new physics patterns.

## 🚀 Quick Start

### Prerequisites
- WSL2 with Ubuntu (Windows users)
- Miniconda installed

### Installation
```bash
# 1. Clone repository
git clone https://github.com/egealkan/particle_accelerator.git
cd particle_accelerator

# 2. Install physics tools (conda)
conda install -c conda-forge root pythia8

# 3. Create Python environment
python3 -m venv .venv
source .venv/bin/activate  # See auto-activation below

# 4. Install Python packages
pip install -r requirements.txt

# 5. Link conda packages to venv
echo "/root/miniconda3/lib/python3.13/site-packages" > .venv/lib/python3.13/site-packages/conda-path.pth

# 6. Test installation
python test_setup.py
```

### Auto-Activate Virtual Environment
Add this to your `~/.bashrc`:
```bash
# Auto-activate venv when entering project directory
cd() {
    builtin cd "$@"
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi
}
```

## 🎯 What This Project Does

- **Simulates** millions of particle collisions using PYTHIA
- **Analyzes** collision data with AI to find particle signatures
- **Discovers** statistical anomalies that could indicate new physics
- **Visualizes** particle tracks in 3D interactive environment
- **Generates** natural language reports of findings

## 🧪 Core Features

### Physics Simulation
- PYTHIA8 collision generation
- ROOT data analysis framework
- Realistic particle interactions

### AI/ML Pipeline
- PyTorch neural networks for particle classification
- Anomaly detection for new physics discovery
- Explainable AI with SHAP/LIME
- NLP-powered analysis reports

### Visualization
- 3D particle track rendering
- Real-time collision monitoring
- Interactive physics parameter controls

## 📊 Project Structure

```
particle_accelerator/
├── README.md
├── requirements.txt
├── test_setup.py           # Installation verification
├── physics/                # Collision simulation
├── ai/                     # ML models
├── visualization/          # 3D rendering
├── data/                   # Collision datasets
└── notebooks/              # Experiments
```

## 🔧 Development

### Run Tests
```bash
python test_setup.py
```

### Start Development
```bash
# Generate collision data
python physics/collision_sim.py

# Train AI models
python ai/classifier.py

# Launch visualization
python visualization/dashboard.py
```

## 📚 Key Technologies

- **PYTHIA8** - Particle collision simulation
- **ROOT** - Data analysis framework  
- **PyTorch** - Deep learning
- **FastAPI** - Web backend
- **Three.js** - 3D visualization

## 🎓 Learning Goals

- Master computational physics simulation
- Apply AI/ML to scientific discovery
- Build interactive scientific applications
- Understand particle physics fundamentals

---

**Ready to discover new physics? Run `python test_setup.py` to verify your installation!** ⚡
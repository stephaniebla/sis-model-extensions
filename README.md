# sis-model-extensions
A simulation-based project that models the spread of infectious diseases using an SIS framework, with control strategies (vaccinations and isolation)).
# 🧪 Epidemic Control Strategy Simulator

This project explores mathematical modeling of infectious disease dynamics using various extensions of the classic SIS (Susceptible-Infectious-Susceptible) model. The models incorporate quarantine, vaccination, and feedback control strategies to simulate the spread and control of diseases in a population.

Built in Python using differential equations and simulated via `solve_ivp`, this tool provides interactive and visual insights into epidemic dynamics and the effects of different control parameters.

---

## 📚 Models Implemented

| Model    | Description |
|----------|-------------|
| `SIS`    | Classic SIS model with basic recovery |
| `SIQS`   | Adds quarantine dynamics |
| `SIVS`   | Adds vaccination as a control strategy |
| `SIQVS`  | Combines quarantine + vaccination + feedback control |

Each model tracks different compartments like Susceptible (S), Infected (I), Quarantined (Q), and Vaccinated (V), depending on the complexity.

---

## 🔬 Mathematical Foundation

The models are based on systems of ODEs. The `SIQVS` model, for instance, uses the following structure:

- \( \frac{dS}{dt} = \lambda - \frac{\beta SI}{N} - (\mu + \psi)S + \chi \phi_1 I + \chi \phi_2 Q \)
- \( \frac{dI}{dt} = \frac{\beta SI}{N} + \frac{\beta \delta IV}{N} - (\mu + \sigma + \phi_1)I \)
- \( \frac{dQ}{dt} = \sigma I - (\phi_2 + \mu)Q \)
- \( \frac{dV}{dt} = \psi S - \frac{\beta \delta IV}{N} + (1 - \chi)\phi_1 I + (1 - \chi)\phi_2 Q - \mu V \)

The basic reproduction number \( R_0 \) is also computed analytically for each model.

---

## 📈 Features

- Multiple epidemic models with increasing complexity
- Interactive sliders for real-time parameter tuning (in `SIQVS`)
- Visualization of population curves over time
- Clean modular code for extension or experimentation

---

## 🛠️ Getting Started

### 📦 Requirements

Install dependencies with:

```bash
pip install numpy matplotlib scipy

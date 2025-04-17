import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider

@dataclass
class ModelParams:
    lambd: float = 3.0
    beta: float = 0.9
    mu: float = 0.01
    phi: float = 0.2
    phi_1: float = 0.2
    phi_2: float = 0.3
    sigma: float = 0.2
    psi: float = 0.4
    chi: float = 1
    delta: float = 0.4
    S0: int = 500
    I0: int = 10
    Q0: int = 0
    V0: int = 0
    T: int = 500

params = ModelParams()


def siqs_ode(t, y):
    S, I, Q = y
    N = S + I
    dS = params.lambd - (params.beta * S * I / N) - params.mu * S + params.phi_1 * I + params.phi_2 * Q
    dI = (params.beta * S * I / N) - (params.mu + params.sigma + params.phi_1) * I
    dQ = params.sigma * I - (params.phi_2 + params.mu) * Q
    return [dS, dI, dQ]


def sis_ode(t, y):
    S, I = y
    N = S + I
    dS = params.lambd - (params.beta * S * I / N) - params.mu * S + params.phi * I
    dI = (params.beta * S * I / N) - (params.mu + params.phi) * I
    return [dS, dI]


def sivs_ode(t, y):
    S, I, V = y
    N = S + I + V
    dS = params.lambd - (params.beta * S * I / N) - (params.mu + params.psi) * S + params.chi * params.phi * I
    dI = (params.beta * S * I / N) + (params.beta * params.delta * I * V / N) - (params.mu + params.phi) * I
    dV = params.psi * S - (params.beta * params.delta * I * V / N) + (1 - params.chi) * params.phi * I - params.mu * V
    return [dS, dI, dV]


def siqvs_ode(t, y):
    S, I, Q, V = y
    N = S + I + V
    dS = params.lambd - (params.beta * S * I / N) - (params.mu + params.psi) * S + params.chi * params.phi_1 * I + params.phi_2 * params.chi * Q
    dI = (params.beta * S * I / N) + (params.beta * params.delta * I * V / N) - (params.mu + params.sigma + params.phi_1) * I
    dQ = params.sigma * I - (params.phi_2 + params.mu) * Q
    dV = params.psi * S - (params.beta * params.delta * I * V / N) + (1 - params.chi) * params.phi_1 * I + (1 - params.chi) * params.phi_2 * Q - params.mu * V
    return [dS, dI, dQ, dV]


def simulate_model(model):
    t_span = (0, params.T)
    t_eval = np.linspace(*t_span, 5000)

    if model == "sis":
        y0 = [params.S0, params.I0]
        sol = solve_ivp(sis_ode, t_span, y0, t_eval=t_eval)
        R0 = params.beta / (params.mu + params.phi)
        return sol.t, sol.y, R0, ["S","I"]
    elif model == "siqs":
        y0 = [params.S0, params.I0, params.Q0]
        sol = solve_ivp(siqs_ode, t_span, y0, t_eval=t_eval)
        R0 = params.beta / (params.mu + params.sigma + params.phi_1)
        return sol.t, sol.y, R0, ["S","I","Q"]
    elif model == "sivs":
        y0 = [params.S0, params.I0, params.V0]
        sol = solve_ivp(sivs_ode, t_span, y0, t_eval=t_eval)
        R0 = (params.beta * (params.mu + params.delta * params.psi)) / ((params.mu + params.psi) * (params.mu + params.phi))
        return sol.t, sol.y, R0, ["S","I","V"]
    elif model == "siqvs":
        y0 = [params.S0, params.I0, params.Q0, params.V0]
        sol = solve_ivp(siqvs_ode, t_span, y0, t_eval=t_eval)
        R0 = (params.beta * (params.mu + (params.delta * params.psi))) / ((params.mu + params.psi) * (params.mu + params.phi_1 + params.sigma))
        return sol.t, sol.y, R0, ["S","I","Q","V"]


def plot_model(time, results, R0=None, title="Model Simulation", comp=["1","2","3","4"]):
    plt.figure(figsize=(10, 6))

    for i in range(len(results)):
        plt.plot(time, results[i], label=f"{comp[i]}")
    if R0 is not None:
        title += f" (R0 = {R0:.2f})"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_with_slider(model):
    if model != "siqvs":
        print("Error: plotting with slider is only possible for siqvs model")
        return
    time, results, R0, comp = simulate_model(model)
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    lines = [ax.plot(time, results[i], label=comp[i])[0] for i in range(len(results))]
    ax.set_title(f"SIQVS Model (R0 = {R0:.2f})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(True)

    ax_sigma = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_sigma = Slider(ax_sigma, 'sigma', 0, 1, valinit=params.sigma)

    ax_psi = plt.axes([0.25, 0.0333, 0.65, 0.03])
    slider_psi = Slider(ax_psi, 'psi', 0, 1, valinit=params.psi)

    ax_delta = plt.axes([0.25, 0.0666, 0.65, 0.03])
    slider_delta = Slider(ax_delta, 'delta', 0, 1, valinit=params.delta)

    def update(val):
        params.sigma = slider_sigma.val
        params.psi = slider_psi.val
        params.delta = slider_delta.val
        time, results, R0 = simulate_model(model)[0:3]
        for i in range(len(results)):
            lines[i].set_ydata(results[i])
        ax.set_title(f"SIQVS Model (R0 = {R0:.2f})")
        fig.canvas.draw_idle()

    slider_sigma.on_changed(update)
    slider_psi.on_changed(update)
    slider_delta.on_changed(update)
    plt.show()



if __name__ == "__main__":
    model = "siqvs"  # Change to "sis", "siqs", "sivs", "siqvs"
    slider = True  # plot with slider, only available for siqvs

    if slider:
        plot_with_slider(model)
    else:
        time, results, R0, comp = simulate_model(model)
        plot_model(time, results, R0=R0, title=f"{model.upper()} Model", comp=comp)

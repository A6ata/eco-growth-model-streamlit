import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

st.set_page_config(page_title="Model zrównoważonego wzrostu", layout="wide")

# =========================
# Model helpers
# =========================

def A(E: float, m: float, Pbar: float) -> float:
    return m * (Pbar - E)


def x_of_tau(tau: float, B: float, delta: float, eta: float) -> float:
    return (delta - (1 - eta) * B * (1 - math.sqrt(tau))) / eta


def theta_of_tau(tau: float, B: float, m: float, Pbar: float) -> float:
    return tau * (Pbar - (B / m) * (tau ** (-0.5) - 1))


def gamma_of_tau(tau: float, B: float, m: float, delta: float, eta: float) -> float:
    x_tau = x_of_tau(tau, B, delta, eta)
    return x_tau / (m + x_tau)


def stationary_E_from_tau(tau: float, B: float, m: float, Pbar: float) -> float:
    return Pbar - (B / m) * (tau ** (-0.5) - 1)


def find_steady_state(B: float, m: float, Pbar: float, delta: float, eta: float):
    tau1 = (B / (Pbar * m + B)) ** 2

    def f(tau: float) -> float:
        return theta_of_tau(tau, B, m, Pbar) - gamma_of_tau(tau, B, m, delta, eta)

    eps = 1e-8
    left = tau1 + eps
    right = 1 - eps

    if left >= right:
        return None

    grid = np.linspace(left, right, 1000)
    vals = np.array([f(t) for t in grid])

    root = None
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        fa, fb = vals[i], vals[i + 1]
        if np.isnan(fa) or np.isnan(fb) or np.isinf(fa) or np.isinf(fb):
            continue
        if fa == 0:
            root = a
            break
        if fa * fb < 0:
            root = brentq(f, a, b, maxiter=200)
            break

    if root is None:
        if abs(f(right)) < 1e-6:
            root = right
        else:
            return None

    tau_star = float(root)
    E_star = stationary_E_from_tau(tau_star, B, m, Pbar)
    x_star = x_of_tau(tau_star, B, delta, eta)
    u_star = math.sqrt(tau_star)
    g_star = (B * (1 - u_star) - delta) / eta
    Z_star = B * (tau_star ** (-0.5) - 1)

    return {
        "tau1": tau1,
        "tau_star": tau_star,
        "E_star": E_star,
        "x_star": x_star,
        "u_star": u_star,
        "g_star": g_star,
        "Z_star": Z_star,
    }


def jacobian(x_star: float, E_star: float, tau_star: float, B: float, m: float, Pbar: float, eta: float) -> np.ndarray:
    F_star = A(E_star, m, Pbar) + B
    return np.array([
        [
            x_star,
            -((1 - eta) / eta) * (1 / E_star) * m * x_star,
            ((1 - eta) / eta) * 0.5 * m * F_star,
        ],
        [
            0.0,
            -m,
            0.5 * F_star / tau_star,
        ],
        [
            tau_star - 1 / E_star,
            x_star / (E_star ** 2),
            (x_star / E_star) * (1 / tau_star),
        ],
    ], dtype=float)


def rhs(t: float, y: np.ndarray, B: float, m: float, Pbar: float, delta: float, eta: float):
    x, E, tau = y

    eps = 1e-10
    E = max(E, eps)
    tau = max(tau, eps)

    emissions = B * (tau ** (-0.5) - 1)
    env_flow = A(E, m, Pbar) - emissions

    dx_over_x = ((1 - eta) / eta) * (1 / E) * env_flow + ((1 - eta) / eta) * B * (1 - math.sqrt(tau)) - delta / eta + x
    dx = x * dx_over_x
    dE = env_flow
    dtau = tau * (m - x / (E * tau) + x)
    return np.array([dx, dE, dtau], dtype=float)


def stop_invalid(t: float, y: np.ndarray, *args):
    x, E, tau = y
    return min(x, E, tau, 1 - tau)

stop_invalid.terminal = True
stop_invalid.direction = -1


def simulate_system(x0, E0, tau0, T, B, m, Pbar, delta, eta):
    sol = solve_ivp(
        rhs,
        (0, T),
        [x0, E0, tau0],
        args=(B, m, Pbar, delta, eta),
        dense_output=False,
        max_step=max(T / 500, 0.05),
        events=stop_invalid,
        rtol=1e-7,
        atol=1e-9,
    )
    return sol


def condition_table(B, m, Pbar, delta, eta, ss):
    rows = []
    cond1 = delta >= (1 - eta) * B
    rows.append({
        "Warunek": r"δ ≥ (1−η)B",
        "Spełniony": "Tak" if cond1 else "Nie",
        "Komentarz": "Zapewnia nieujemność x(τ) w analizowanym przedziale.",
    })

    cond2 = Pbar > delta / (eta * m + delta)
    rows.append({
        "Warunek": r"P̄ > δ / (ηm + δ)",
        "Spełniony": "Tak" if cond2 else "Nie",
        "Komentarz": "Warunek istnienia i jednoznaczności dodatniego punktu stacjonarnego.",
    })

    if ss is not None:
        tau_star = ss["tau_star"]
        cond3 = B * (1 - math.sqrt(tau_star)) > delta
        cond4 = delta > (1 - eta) * B * (1 - math.sqrt(tau_star))
        rows.append({
            "Warunek": r"B(1−√τ*) > δ",
            "Spełniony": "Tak" if cond3 else "Nie",
            "Komentarz": "Warunek dodatniego długookresowego tempa wzrostu.",
        })
        rows.append({
            "Warunek": r"δ > (1−η)B(1−√τ*)",
            "Spełniony": "Tak" if cond4 else "Nie",
            "Komentarz": "Warunek zbieżności funkcji celu na ścieżce zrównoważonego wzrostu.",
        })
    else:
        rows.append({
            "Warunek": r"B(1−√τ*) > δ",
            "Spełniony": "—",
            "Komentarz": "Brak punktu stacjonarnego, więc warunek nie został obliczony.",
        })
        rows.append({
            "Warunek": r"δ > (1−η)B(1−√τ*)",
            "Spełniony": "—",
            "Komentarz": "Brak punktu stacjonarnego, więc warunek nie został obliczony.",
        })

    return pd.DataFrame(rows)


# =========================
# UI
# =========================

st.title("Model optymalnego zrównoważonego wzrostu")
st.markdown(
    """
Aplikacja wizualizuje model Cazzavillana–Musu z dynamicznymi zmiennymi:
- **x = C/K** — relacja konsumpcji do kapitału,
- **E** — zasób środowiska,
- **τ** — względna wartość środowiska względem kapitału.

Poniżej możesz zmieniać parametry modelu, sprawdzać warunki istnienia rozwiązania,
wyznaczać punkt stacjonarny, analizować lokalną stabilność i symulować trajektorie dynamiczne.
"""
)

with st.sidebar:
    st.header("Parametry modelu")

    B = st.slider("B — produktywność kapitału", min_value=0.1, max_value=5.0, value=1.2, step=0.05)
    m = st.slider("m — tempo asymilacji środowiska", min_value=0.05, max_value=2.0, value=0.3, step=0.01)
    Pbar = st.slider("P̄ — maksymalna pojemność środowiska", min_value=0.1, max_value=3.0, value=1.2, step=0.05)
    delta = st.slider("δ — stopa dyskonta", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
    eta = st.slider("η — awersja / odwrotność EIS", min_value=0.1, max_value=3.0, value=1.2, step=0.05)

    st.divider()
    st.header("Symulacja")
    T = st.slider("Horyzont czasu T", min_value=5, max_value=300, value=80, step=5)
    use_steady_defaults = st.checkbox("Ustaw warunki początkowe blisko stanu ustalonego", value=True)

ss = find_steady_state(B, m, Pbar, delta, eta)
cond_df = condition_table(B, m, Pbar, delta, eta, ss)

st.header("1. Warunki istnienia ścieżki zrównoważonego wzrostu")
col_a, col_b = st.columns([1.2, 1])
with col_a:
    st.dataframe(cond_df, use_container_width=True, hide_index=True)
with col_b:
    if ss is None:
        st.error("Dla wybranych parametrów nie znaleziono dodatniego punktu stacjonarnego w przedziale τ ∈ (τ₁, 1).")
    else:
        st.success("Dla wybranych parametrów znaleziono punkt stacjonarny modelu.")
        st.write(f"**τ₁ =** {ss['tau1']:.6f}")
        st.write(f"**τ* =** {ss['tau_star']:.6f}")

if ss is not None:
    st.header("2. Stan ustalony")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("x*", f"{ss['x_star']:.4f}")
    c2.metric("E*", f"{ss['E_star']:.4f}")
    c3.metric("τ*", f"{ss['tau_star']:.4f}")
    c4.metric("u* = √τ*", f"{ss['u_star']:.4f}")
    c5.metric("g*", f"{ss['g_star']:.4f}")

    # Theta vs Gamma
    tau1 = ss["tau1"]
    tau_grid = np.linspace(tau1 + 1e-5, 0.999999, 800)
    theta_vals = [theta_of_tau(t, B, m, Pbar) for t in tau_grid]
    gamma_vals = [gamma_of_tau(t, B, m, delta, eta) for t in tau_grid]

    fig_intersection = go.Figure()
    fig_intersection.add_trace(go.Scatter(x=tau_grid, y=theta_vals, mode="lines", name="Θ(τ)"))
    fig_intersection.add_trace(go.Scatter(x=tau_grid, y=gamma_vals, mode="lines", name="Γ(τ)"))
    fig_intersection.add_vline(x=ss["tau_star"], line_dash="dash", annotation_text="τ*", annotation_position="top")
    fig_intersection.add_trace(go.Scatter(x=[ss["tau_star"]], y=[theta_of_tau(ss["tau_star"], B, m, Pbar)], mode="markers", name="Punkt przecięcia"))
    fig_intersection.update_layout(
        title="Przecięcie funkcji Θ(τ) i Γ(τ)",
        xaxis_title="τ",
        yaxis_title="Wartość funkcji",
        height=500,
    )
    st.plotly_chart(fig_intersection, use_container_width=True)

    st.header("3. Stabilność lokalna")
    J = jacobian(ss["x_star"], ss["E_star"], ss["tau_star"], B, m, Pbar, eta)
    eigvals = np.linalg.eigvals(J)
    trace = float(np.trace(J))
    det = float(np.linalg.det(J))
    neg_count = int(np.sum(np.real(eigvals) < 0))
    pos_count = int(np.sum(np.real(eigvals) > 0))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Macierz Jacobiego w stanie ustalonym**")
        st.dataframe(pd.DataFrame(J, index=["dx/dx", "dE/dx", "dτ/dx"], columns=["x", "E", "τ"]), use_container_width=True)
    with col2:
        eig_df = pd.DataFrame({
            "Wartość własna": [f"λ{i+1}" for i in range(len(eigvals))],
            "Część rzeczywista": np.real(eigvals),
            "Część urojona": np.imag(eigvals),
        })
        st.dataframe(eig_df, use_container_width=True, hide_index=True)
        st.write(f"**Ślad J:** {trace:.6f}")
        st.write(f"**Wyznacznik J:** {det:.6f}")
        if neg_count == 1 and pos_count == 2:
            st.success("Punkt stacjonarny ma charakter siodłowy: 1 wartość własna o ujemnej i 2 o dodatniej części rzeczywistej.")
        else:
            st.warning("Konfiguracja wartości własnych odbiega od klasycznego przypadku saddle-path.")

    st.header("4. Symulacja dynamiczna")

    with st.sidebar:
        if use_steady_defaults:
            x0_default = max(1e-4, ss["x_star"] * 1.05)
            E0_default = max(1e-4, ss["E_star"] * 0.95)
            tau0_default = min(0.999, max(ss["tau1"] + 1e-4, ss["tau_star"] * 1.05))
        else:
            x0_default = max(0.01, ss["x_star"] if ss["x_star"] > 0 else 0.1)
            E0_default = max(0.01, ss["E_star"] if ss["E_star"] > 0 else 0.1)
            tau0_default = min(0.99, max(ss["tau1"] + 1e-3, ss["tau_star"] if ss["tau_star"] > 0 else 0.5))

        x0 = st.number_input("x₀", min_value=0.0001, max_value=20.0, value=float(x0_default), step=0.01, format="%.4f")
        E0 = st.number_input("E₀", min_value=0.0001, max_value=20.0, value=float(E0_default), step=0.01, format="%.4f")
        tau0 = st.number_input("τ₀", min_value=float(ss["tau1"] + 1e-4), max_value=0.9999, value=float(tau0_default), step=0.01, format="%.4f")

    sol = simulate_system(x0, E0, tau0, T, B, m, Pbar, delta, eta)

    if len(sol.t) < 2:
        st.error("Symulacja zatrzymała się natychmiast — sprawdź warunki początkowe albo zakres parametrów.")
    else:
        t = sol.t
        x_path, E_path, tau_path = sol.y
        u_path = np.sqrt(np.maximum(tau_path, 1e-12))
        Z_path = B * (np.maximum(tau_path, 1e-12) ** (-0.5) - 1)

        tabs = st.tabs(["Ścieżki czasowe", "Portret fazowy", "Tabela wyników"])

        with tabs[0]:
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=t, y=x_path, mode="lines", name="x(t)"))
            fig_time.add_trace(go.Scatter(x=t, y=E_path, mode="lines", name="E(t)"))
            fig_time.add_trace(go.Scatter(x=t, y=tau_path, mode="lines", name="τ(t)"))
            fig_time.add_trace(go.Scatter(x=t, y=u_path, mode="lines", name="u(t)=√τ(t)"))
            fig_time.add_trace(go.Scatter(x=t, y=Z_path, mode="lines", name="Z(t)"))
            fig_time.update_layout(title="Ścieżki czasowe zmiennych modelu", xaxis_title="t", yaxis_title="Wartość", height=550)
            st.plotly_chart(fig_time, use_container_width=True)

        with tabs[1]:
            p1, p2 = st.columns(2)
            with p1:
                fig_phase_1 = go.Figure()
                fig_phase_1.add_trace(go.Scatter(x=E_path, y=tau_path, mode="lines", name="Trajektoria"))
                fig_phase_1.add_trace(go.Scatter(x=[E0], y=[tau0], mode="markers", name="Start"))
                fig_phase_1.add_trace(go.Scatter(x=[ss["E_star"]], y=[ss["tau_star"]], mode="markers", name="Stan ustalony"))
                fig_phase_1.update_layout(title="Portret fazowy: (E, τ)", xaxis_title="E", yaxis_title="τ", height=450)
                st.plotly_chart(fig_phase_1, use_container_width=True)
            with p2:
                fig_phase_2 = go.Figure()
                fig_phase_2.add_trace(go.Scatter(x=E_path, y=x_path, mode="lines", name="Trajektoria"))
                fig_phase_2.add_trace(go.Scatter(x=[E0], y=[x0], mode="markers", name="Start"))
                fig_phase_2.add_trace(go.Scatter(x=[ss["E_star"]], y=[ss["x_star"]], mode="markers", name="Stan ustalony"))
                fig_phase_2.update_layout(title="Portret fazowy: (E, x)", xaxis_title="E", yaxis_title="x", height=450)
                st.plotly_chart(fig_phase_2, use_container_width=True)

        with tabs[2]:
            result_df = pd.DataFrame({
                "t": t,
                "x": x_path,
                "E": E_path,
                "tau": tau_path,
                "u": u_path,
                "Z": Z_path,
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Pobierz wyniki CSV", data=csv, file_name="wyniki_symulacji.csv", mime="text/csv")
else:
    st.info("Zmień parametry tak, aby model miał dodatni punkt stacjonarny. Wtedy pojawią się sekcje stanu ustalonego, stabilności i symulacji.")

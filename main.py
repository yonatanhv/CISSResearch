import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def hamiltonianOperator2D(psi, dx, dy, V):
    laplacian_x = (np.roll(psi, -1, axis=1) - 2 * psi + np.roll(psi, 1, axis=1)) / dx ** 2
    laplacian_y = (np.roll(psi, -1, axis=0) - 2 * psi + np.roll(psi, 1, axis=0)) / dy ** 2
    laplacian = laplacian_x + laplacian_y
    return -0.5 * laplacian + V * psi


def createWavePacket2D(X, Y, x0, y0, sigma_x, sigma_y, ky, m, dx, dy):
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    phi = np.arctan2(Y - y0, X - x0)

    gaussian_envelope = np.exp(-((X - x0) ** 2) / (2 * sigma_x ** 2)) * np.exp(-((Y - y0) ** 2) / (2 * sigma_y ** 2))
    angular_momentum_term = (r ** np.abs(m)) * np.exp(1j * m * phi)
    plane_wave = np.exp(1j * ky * Y)

    psi = angular_momentum_term * gaussian_envelope * plane_wave
    probability_sum = np.sum(np.abs(psi) ** 2) * dx * dy
    psi /= np.sqrt(probability_sum)

    return psi


def timeDerivative2D(t, psi, dx, dy, V):
    return -1j * hamiltonianOperator2D(psi, dx, dy, V)


def RK4_2D(t, psi, dx, dy, dt, V):
    k1 = timeDerivative2D(t, psi, dx, dy, V) * dt
    k2 = timeDerivative2D(t + 0.5 * dt, psi + 0.5 * k1, dx, dy, V) * dt
    k3 = timeDerivative2D(t + 0.5 * dt, psi + 0.5 * k2, dx, dy, V) * dt
    k4 = timeDerivative2D(t + dt, psi + k3, dx, dy, V) * dt
    psi_next = psi + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return psi_next


def LJ_wall(X, Y, y_wall_pos, epsilon, sigma, Vmax):
    V = np.zeros_like(X)
    mask = Y < y_wall_pos
    dy_wall = y_wall_pos - Y[mask]
    V[mask] = 4 * epsilon * ((sigma / dy_wall) ** 12 - (sigma / dy_wall) ** 6)
    V[~mask] = Vmax
    V = np.clip(V, -epsilon, Vmax)
    return V


def scatterer_potential(X, Y, y_c, A_au, b_au):
    r = np.sqrt(X ** 2 + (Y - y_c) ** 2)
    V_ws = A_au * np.exp(-r / b_au)
    return V_ws


def calc_quantum_force(psi, X, Y, y_c, dx, dy, A_ev=200.0, b_ang=1.0):
    A_au = A_ev / 27.2114
    b_au = b_ang / 0.52918
    r = np.sqrt(X ** 2 + (Y - y_c) ** 2)
    r_safe = np.where(r == 0, 1e-10, r)
    force_matrix = -A_au * np.exp(-r_safe / b_au) * ((Y - y_c) / (b_au * r_safe))
    prob_density = np.abs(psi) ** 2
    expected_force = np.sum(prob_density * force_matrix) * dx * dy
    return expected_force


def Animation2D_WithChecks(psi_init, dx, dy, dt, V_wall, X, Y):
    fig, ax = plt.subplots(figsize=(8, 8))

    m_c = 0.2
    k_spring = 1.0
    y_eq = 1.0

    A_au = 200.0 / 27.2114
    b_au = 0.3 / 0.52918
    X_sq = X ** 2

    y_c_current = y_eq
    y_c_prev = y_eq

    state = {'psi': psi_init, 't': 0.0, 'y_c_curr': y_c_current, 'y_c_prev': y_c_prev}
    prob_density = np.abs(state['psi']) ** 2
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    img = ax.imshow(prob_density, extent=extent, origin='lower',
                    cmap='magma', animated=True, vmax=np.max(prob_density))

    levels = [1.0, 10.0, 50.0, 100.0]
    ax.contour(X, Y, V_wall, levels=levels, colors='white', alpha=0.3, linewidths=1.5, linestyles='dashed')

    scatter_marker, = ax.plot([0], [y_c_current], 'o', markerfacecolor='none',
                              markeredgecolor='green', markersize=14, markeredgewidth=2)

    ax.set_xlabel("X (a.u.)")
    ax.set_ylabel("Y (a.u.)")
    ax.set_title("Wavepacket driving a Classical Oscillator")
    fig.colorbar(img, ax=ax, label="Probability Density |ψ|²")

    initial_norm = np.sum(np.abs(psi_init) ** 2) * dx * dy

    telemetry_text = ax.text(0.03, 0.97, '', transform=ax.transAxes,
                             color='white', fontsize=10, family='monospace',
                             verticalalignment='top',
                             bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=4))

    def update(frame):
        steps_per_frame = 250

        for _ in range(steps_per_frame):
            dy_c = Y - state['y_c_curr']
            r = np.sqrt(X_sq + dy_c ** 2)
            r_safe = r + 1e-10

            V_ws = A_au * np.exp(-r / b_au)
            V_current = V_wall + V_ws

            force_matrix = -V_ws * (dy_c / (b_au * r_safe))

            prob_density = np.abs(state['psi']) ** 2
            quantum_force = np.sum(prob_density * force_matrix) * dx * dy

            spring_force = -k_spring * (state['y_c_curr'] - y_eq)
            acceleration = (spring_force + quantum_force) / m_c

            y_c_next = 2 * state['y_c_curr'] - state['y_c_prev'] + acceleration * dt ** 2
            state['y_c_prev'] = state['y_c_curr']
            state['y_c_curr'] = y_c_next

            state['psi'] = RK4_2D(state['t'], state['psi'], dx, dy, dt, V_current)
            state['t'] += dt

        psi_current = state['psi']
        current_norm = np.sum(np.abs(psi_current) ** 2) * dx * dy
        delta_norm = current_norm - initial_norm

        img.set_data(np.abs(psi_current) ** 2)
        scatter_marker.set_data([0], [state['y_c_curr']])

        telemetry_text.set_text(f"t   : {state['t']:.3f}\nΔN  : {delta_norm:+.2e} ")

        return [img, scatter_marker, telemetry_text]

    print("Generating Optimized Engine Animation...")
    ani = FuncAnimation(fig, update, frames=400, interval=30, blit=True)
    plt.show()


Resolution = 10
L_limit = 4.0
numPoints = int(2 * L_limit * Resolution)

x_arr = np.linspace(-L_limit, L_limit, numPoints)
y_arr = np.linspace(-L_limit, L_limit, numPoints)

X, Y = np.meshgrid(x_arr, y_arr)
dx = x_arr[1] - x_arr[0]
dy = y_arr[1] - y_arr[0]

dt = 0.00002

x0, y0 = 0.0, -2.0
sigma_x, sigma_y = 0.75, 0.75
ky = 5.0
m_quantum = 0

psi = createWavePacket2D(X, Y, x0, y0, sigma_x, sigma_y, ky, m_quantum, dx, dy)

y_wall = 3.5
epsilon_w = 2.0
sigma_w = 0.5
V_max = 150.0

V_wall = LJ_wall(X, Y, y_wall, epsilon_w, sigma_w, V_max)

Animation2D_WithChecks(psi, dx, dy, dt, V_wall, X, Y)
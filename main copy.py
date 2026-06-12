import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

#Opertaors
#def momentumOperator(psi, dx): this function is for closed language barriers
    #p_psi = np.zeros_like(psi, dtype=complex)
    #p_psi[1:-1] = -1j * (psi[2:]-psi[:-2])/(2*dx)
    #return p_psi

#def hamiltonianOperator(psi, dx): this function is for closed language barriers
    #H_psi = np.zeros_like(psi, dtype=complex)
    #H_psi[1:-1] = -0.5*(psi[2:] - 2*psi[1:-1] + psi[:-2]) / (dx**2)
    #return H_psi

def momentumOperator(psi, dx):
    return -1j * (np.roll(psi, -1) - np.roll(psi, 1))/(2*dx)

def hamiltonianOperator(psi, dx):
    laplacian = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1))/dx**2
    return -0.5*laplacian

#Checks
def checkFunctionByProbability(probability, dx):
    functionStabilityCheck = np.sum(probability) * dx
    return functionStabilityCheck

def checkFunctionByEnergy(psi, dx):
    p_psi = momentumOperator(psi, dx)
    H_psi = hamiltonianOperator(psi, dx)

    expected_p = np.sum(np.conj(psi) * p_psi)*dx
    expected_E = np.sum(np.conj(psi) * H_psi)*dx

    print(f"Expected Momentum <p>: {np.real(expected_p):.4f}")
    print(f"Expected Energy <E>:  {np.real(expected_E):.4f}")

def runChecks(psi, dx, probability):
    print(f"Probability check: {checkFunctionByProbability(probability, dx)}")
    checkFunctionByEnergy(psi, dx)

def compareResolutions(res_list):
    plt.figure(figsize=(10, 6))

    # הגדרת dt קבוע וזמן ריצה כולל
    fixed_dt = 0.0002
    total_steps = 5000

    for res in res_list:
        # 1. הגדרת גריד ספציפי
        current_numPoints = int((endPoint - startPoint) * res)
        x_local = np.linspace(startPoint, endPoint, current_numPoints)
        dx_local = x_local[1] - x_local[0]

        # 2. אתחול פונקציית הגל ונרמול
        psi_local = (np.exp(-(x_local - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x_local))
        psi_local /= np.sqrt(np.sum(np.abs(psi_local) ** 2) * dx_local)

        trace_t, trace_norm_dev = [], []
        t_local = 0.0

        # 3. הרצה
        for step in range(total_steps):
            psi_local = RK4(t_local, psi_local, dx_local, fixed_dt)
            t_local += fixed_dt

            if step % 20 == 0:
                # חישוב הנרמול הנוכחי והסטייה מהיעד (1.0)
                current_n = np.sum(np.abs(psi_local) ** 2) * dx_local
                trace_t.append(t_local)
                trace_norm_dev.append(current_n - 1.0)

        plt.plot(trace_t, trace_norm_dev, label=f'Res = {res} (dx={dx_local:.4f})')

    # עיצוב הגרף
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.title(f"Normalization Stability Study\n(Fixed dt = {fixed_dt:.2e})")
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Δ Probability (Norm - 1.0)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def compareTimeSteps(dt_list, fixed_res=100):
    plt.figure(figsize=(10, 6))
    numPoints_local = int((endPoint - startPoint) * fixed_res)
    x_local = np.linspace(startPoint, endPoint, numPoints_local)
    dx_local = x_local[1] - x_local[0]

    total_sim_time = 0.1  # זמן סימולציה כולל קבוע

    for dt_val in dt_list:
        psi_local = (np.exp(-(x_local - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x_local))
        psi_local /= np.sqrt(np.sum(np.abs(psi_local) ** 2) * dx_local)

        trace_t, trace_norm = [], []
        t_local = 0.0
        steps = int(total_sim_time / dt_val)

        for step in range(steps):
            psi_local = RK4(t_local, psi_local, dx_local, dt_val)
            t_local += dt_val
            if step % 5 == 0:
                trace_t.append(t_local)
                trace_norm.append((np.sum(np.abs(psi_local) ** 2) * dx_local) - 1.0)

        plt.plot(trace_t, trace_norm, label=f'dt = {dt_val:.2e}')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(f"Normalization vs Time Step (Fixed Res = {fixed_res})")
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Δ Probability")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


def normalizationWithDifferentRes(res_list):
    plt.figure(figsize=(10, 6))

    fixed_dt = 0.0001
    total_steps = 4000

    for res in res_list:
        numP = int((endPoint - startPoint) * res)
        x_l = np.linspace(startPoint, endPoint, numP, endpoint=False)
        dx_l = x_l[1] - x_l[0]

        psi_l = (np.exp(-(x_l - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x_l))
        psi_l /= np.sqrt(np.sum(np.abs(psi_l) ** 2) * dx_l)

        trace_t, trace_norm_dev = [], []
        t_l = 0.0

        for step in range(total_steps):
            psi_l = RK4(t_l, psi_l, dx_l, fixed_dt)
            t_l += fixed_dt

            if step % 20 == 0:
                current_norm = np.sum(np.abs(psi_l) ** 2) * dx_l
                trace_t.append(t_l)
                trace_norm_dev.append(current_norm - 1.0)

        plt.plot(trace_t, trace_norm_dev, label=f'Res {res} (pts={numP})', alpha=0.8)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.title(f"Normalization Fluctuations vs Grid Resolution\n(Fixed dt = {fixed_dt})")
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Δ Probability (Norm - 1.0)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


#Time continuing functions
def timeDerivative(t, psi, dx):
    return -1j*hamiltonianOperator(psi, dx)

def RK4(t, psi, dx, dt):
    k1 = timeDerivative(t, psi, dx)*dt
    k2 = timeDerivative(t + 0.5*dt, psi + 0.5*k1, dx)*dt
    k3 = timeDerivative(t + 0.5*dt, psi + 0.5*k2, dx)*dt
    k4 = timeDerivative(t + dt, psi + k3, dx)*dt
    psi_next = psi + (k1 + 2*k2 + 2*k3 + k4)/6.0
    return psi_next


def Animation(t, psi, dx, dt):
    stepsAmount = 1000000
    plot_every = 250
    plt.ion()
    fig, ax = plt.subplots()
    realLine, = ax.plot(x, np.real(psi), label='Real part of ψ')
    probLine, = ax.plot(x, np.abs(psi) ** 2, label='Probability (|ψ|^2)', color='black')
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(startPoint, endPoint)
    ax.legend(loc='upper right')
    for step in range(stepsAmount):
        psi = RK4(t, psi, dx, dt)
        t += dt
        if step % plot_every == 0:
            realLine.set_ydata(np.real(psi))
            probLine.set_ydata(np.abs(psi) ** 2)
            plt.pause(0.001)
    plt.ioff()
    plt.show()


def plotChecks(t, psi, dx, dt):
    stepsAmount = 10000
    trace_t, trace_norm, trace_energy = [], [], []

    start_time = time.time()

    for step in range(stepsAmount):
        psi = RK4(t, psi, dx, dt)
        t += dt
        if step % 10 == 0:
            current_norm = np.sum(np.abs(psi) ** 2) * dx
            H_psi = hamiltonianOperator(psi, dx)
            current_energy = np.real(np.sum(np.conj(psi) * H_psi) * dx)
            trace_t.append(t)
            trace_norm.append(current_norm)
            trace_energy.append(current_energy)

    end_time = time.time()
    runtime = end_time - start_time

    avg_norm = np.mean(trace_norm)
    avg_energy = np.mean(trace_energy)

    norm_dev = (np.array(trace_norm) - 1.0)
    energy_dev = (np.array(trace_energy) - avg_energy)

    m_norm, _ = np.polyfit(trace_t, trace_norm, 1)
    m_energy, _ = np.polyfit(trace_t, trace_energy, 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    fig.suptitle(f"Simulation Diagnostics (Resolution = {Resolution})\nTotal Run Time: {runtime:.2f} seconds",fontsize=14, fontweight='bold')

    ax1.plot(trace_t, norm_dev, color='blue', linewidth=0.8)
    ax1.set_title(f"Normalization Stability (Avg Norm = {avg_norm:.8f})\nSlope (m) = {m_norm:.2e}")
    ax1.set_ylabel("Δ Probability (Noise)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.plot(trace_t, energy_dev, color='red', linewidth=0.8)
    ax2.set_title(f"Energy Conservation (Avg E = {avg_energy:.7f})\nSlope (m) = {m_energy:.2e}")
    ax2.set_ylabel("Δ Energy (a.u.)")
    ax2.set_xlabel("Time (a.u.)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    plt.show(block=True)

#Optimaztion
def optimizingValues(res_start, res_end, res_jump,
                     dt_start, dt_end, dt_jump,
                     goal_norm, goal_energy):
    valid_configs = []
    stepAmount = 500
    goal_norm*=10
    goal_energy *= 10

    for res in range(res_start, res_end + 1, res_jump):
        numP = int((endPoint - startPoint) * res)
        x_test = np.linspace(startPoint, endPoint, numP)
        dx_test = x_test[1] - x_test[0]

        for dt_val in np.arange(dt_start, dt_end, dt_jump):
            if dt_val > (dx_test ** 2) * 1.2:
                continue

            psi_test = (np.exp(-(x_test - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x_test))
            psi_test /= (np.sqrt(np.sum(np.abs(psi_test) ** 2) * dx_test))

            H_psi_init = hamiltonianOperator(psi_test, dx_test)
            e_init = np.real(np.sum(np.conj(psi_test) * H_psi_init) * dx_test)

            max_dn = 0
            max_de = 0
            stable = True

            for _ in range(stepAmount):
                psi_test = RK4(0, psi_test, dx_test, dt_val)

                if np.isnan(psi_test[0]):
                    stable = False
                    break

                curr_n = np.sum(np.abs(psi_test) ** 2) * dx_test
                dn = abs(curr_n - 1.0)

                H_psi = hamiltonianOperator(psi_test, dx_test)
                curr_e = np.real(np.sum(np.conj(psi_test) * H_psi) * dx_test)
                de = abs(curr_e - e_init)

                max_dn = max(max_dn, dn)
                max_de = max(max_de, de)

                if dn > 0.05:
                    stable = False
                    break

            if stable and goal_norm/10 <= max_dn <= goal_norm and goal_energy/10 <= max_de <= goal_energy:
                valid_configs.append((res, dt_val, max_dn, max_de))
    return valid_configs

def optimizedSetup(target_norm, target_energy):
    global Resolution, dt, x, dx, psi
    results = optimizingValues(
        res_start=15, res_end=100, res_jump=5,
        dt_start=0.0001, dt_end=0.05, dt_jump=0.0005,
        goal_norm=target_norm, goal_energy=target_energy
    )

    print(f"Found {len(results)} valid configurations:")
    for r, d, n, e in results:
        print(f"Res: {r}, dt: {d:.5f} | Norm Err: {n:.2e}, Energy Err: {e:.2e}")

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        best_res, best_dt, _, _ = results[0]

        Resolution = best_res
        dt = best_dt

        numPoints = int((endPoint - startPoint) * Resolution)
        x = np.linspace(startPoint, endPoint, numPoints)
        dx = x[1] - x[0]

        psi = (np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * k0 * x))
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

        print(f"Running diagnostics with Res: {Resolution}, dt: {dt}")
        plotChecks(0.0, psi, dx, dt)

Resolution = 15
startPoint = -50
endPoint = -startPoint
numPoints = -1 * (startPoint-endPoint) * Resolution
x = np.linspace(startPoint,endPoint, numPoints, endpoint=False)

#intializing wave packet
x0=0
k0 = 5.0
dx = x[1] - x[0]
#dt = (dx**2)*0.5
dt = 0.0001
sigma = 1.0
psi = (np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j * k0 * x))
probability = np.abs(psi)**2

#normalizing
normalization = np.sqrt(probability.sum() * dx)
psi = psi/normalization
probability = np.abs(psi)**2

#main
#runChecks(psi, dx, probability)
#Animation(0.0, psi, dx, dt)
#plotChecks(0.0, psi, dx, dt)
#compareResolutions([5,10,15,20,25])
#compareTimeSteps([0.0001, 0.00005, 0.00001])

#optimizedSetup(1e-16,1e-15)
#Animation(0.0, psi, dx, dt)
#plotChecks(0.0, psi, dx, dt)

normalizationWithDifferentRes([10,20,50,100])
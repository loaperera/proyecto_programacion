# proyecto_programacion

# APARTADO 2
from numpy import array, polyfit, polyval, linspace, diag, transpose, matmul
from numpy.linalg import svd, solve  
import matplotlib.pyplot as plt

### Método para ajuste lineal por numpy
def Test_numpy_regression():
    q_tunel = array([500, 1000, 1500, 2000, 2500, 3000])
    F_drag_tunel = array([12.1, 25.3, 37.8, 51.0, 62.5, 76.2])
    c = polyfit(q_tunel, F_drag_tunel, deg=1)
    CDA = c[0]
    print("Drag coefficient CDA:", CDA)

    xp = linspace(0, 3000, 100)
    yp = CDA * xp
    plt.scatter(q_tunel, F_drag_tunel, label='Data')
    plt.plot(xp, yp, 'r', label='Linear regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Test_numpy_regression()

### Método para ajuste lineal manual
def Vandermonde(x, N, M):
    return array([[x[j]**i for i in range(1, N+1)] for j in range(M)])

def regression_vandermonde(x, y, N):
    M = len(x)
    A = Vandermonde(x, N, M)
    return solve(matmul(transpose(A), A), matmul(transpose(A), y))

def polynomial(w, x):
    N = len(w)
    return sum(w[k] * x**(k+1) for k in range(N))


def Test_manual_regression():
    q_tunel = array([500, 1000, 1500, 2000, 2500, 3000])
    F_drag_tunel = array([12.1, 25.3, 37.8, 51.0, 62.5, 76.2])
    w = regression_vandermonde(q_tunel, F_drag_tunel, 1)
    CDA = w[0]
    print("Drag coefficient CDA:", CDA)
    xp = array([500, 1000, 1500, 2000, 2500, 3000])
    yp = polynomial(w, xp)
    plt.scatter(q_tunel, F_drag_tunel, label='Data')
    plt.plot(xp, yp, 'r', label='Linear regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Test_manual_regression()

# APARTADO 3
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt

def estimar_densidad_local():
### Datos de Partida: Atmósfera Estándar (ISA)
    altitud_isa = np.array([0, 500, 1000, 1500, 2000, 2500])
    densidad_isa = np.array([1.225, 1.167, 1.112, 1.058, 1.007, 0.957])
    
### Altitud de apogeo registrada por el radar
    h_apogeo = 405.0 
    
### Creación del polinomio interpolante de Lagrange (estilo week10.py)
    polinomio_densidad = lagrange(altitud_isa, densidad_isa)
    
### Evaluación del polinomio en h = 405 m
    rho_apogeo = polinomio_densidad(h_apogeo)
    
### Resultados e interpretación física
    print(f"--- Apartado 3: Modelo Atmosférico Local ---")
    print(f"Densidad del aire estimada a {h_apogeo} m: {rho_apogeo:.4f} kg/m^3")
    print("Sentido Físico: Como era de esperar, al ascender a 405 metros, la densidad del aire experimenta una disminución respecto al nivel del mar (1.225 kg/m^3), lo que se traducirá en una menor resistencia aerodinámica sobre la sonda en su punto de apogeo.")
    
### Gráfica del interpolante
    hp = np.linspace(0, 2500, 100)
    rhop = polinomio_densidad(hp)
    
    plt.scatter(altitud_isa, densidad_isa, label='Datos ISA')
    plt.plot(hp, rhop, 'r', label='Polinomio interpolante')
    plt.scatter([h_apogeo], [rho_apogeo], color='green', zorder=5, label=f'Apogeo ({h_apogeo} m)')
    plt.xlabel('Altitud h (m)')
    plt.ylabel('Densidad ρ (kg/m³)')
    plt.title('Interpolación 1D - Densidad ISA')
    plt.legend()
    plt.show()
    
    return rho_apogeo

if __name__ == "__main__":
    rho_estimada = estimar_densidad_local()

# APARTADO 4
from numpy import zeros, linspace, array, sqrt
import matplotlib.pyplot as plt

### Derivada numérica (diferencias finitas).

def Derivatives(t, f):

    N = len(t) - 1
    dt = t[1] - t[0]
    df = zeros(N+1)

    df[0] = ( -3*f[0] + 4*f[1] - f[2] ) / ( 2*dt )

    for i in range(1, N):
        df[i] = ( f[i+1] - f[i-1] ) / ( 2*dt )

    df[N] = ( 3*f[N] - 4*f[N-1] + f[N-2] ) / ( 2*dt )

    return df

### Velocidades a partir de telemetría del radar.

def Radar_kinematics(t, x, y):

    Vx = Derivatives(t, x)
    Vy = Derivatives(t, y)
    V  = sqrt( Vx**2 + Vy**2 )

    return Vx, Vy, V

### Gráfica y resultados

def Plot_Apartado4(ax, t, x, y):

    Vx, Vy, V = Radar_kinematics(t, x, y)

    print("\n" + "="*54)
    print("         APARTADO 4: CINEMÁTICA DEL RADAR")
    print("="*54)
    print(f"{'t (s)':>8} | {'Vx (m/s)':>12} | {'Vy (m/s)':>12} | {'V (m/s)':>12}")
    print("-"*54)

    for i in range(len(t)):
        print(f"{t[i]:8.2f} | {Vx[i]:12.2f} | {Vy[i]:12.2f} | {V[i]:12.2f}")

    print("="*54)
    print("Velocidad inicial =", round(V[0], 2), "m/s")
    print("Velocidad en apogeo =", round(V[9], 2), "m/s")
    print("Velocidad final =", round(V[-1], 2), "m/s")
    print("Sentido físico: Vx disminuye por la resistencia del aire y Vy es 0 en el apogeo (t=4.5 s).\n")

    ax.clear()
    ax.plot(t, Vx, ".-", label="Vx")
    ax.plot(t, Vy, ".-", label="Vy")
    ax.plot(t, V,  ".-", label="V")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Velocidad (m/s)")
    ax.set_title("Apartado 4: Cinemática del Radar")
    ax.legend()
    ax.grid()

    texto = f"V0 = {V[0]:.2f} m/s | V_apogeo = {V[9]:.2f} m/s | Vf = {V[-1]:.2f} m/s\nSentido físico: Vx baja por el drag y Vy=0 en el apogeo."
    ax.text(0.05, 0.05, texto, transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8))


if __name__ == "__main__":

    t_tel = linspace(0, 8, 17)
    x_tel = array([0, 48, 93, 137, 178, 218, 256, 292, 326, 359, 390, 419, 448, 474, 500, 524, 547])
    y_tel = array([0, 85, 160, 225, 280, 325, 360, 385, 400, 405, 400, 385, 360, 325, 280, 225, 160])

    fig, ax = plt.subplots()
    Plot_Apartado4(ax, t_tel, x_tel, y_tel)
    plt.show()

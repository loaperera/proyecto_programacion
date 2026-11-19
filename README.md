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
from numpy import array, linspace, zeros, dot
import matplotlib.pyplot as plt

def Lagrange_polynomials(x, xp):

###Evalúa los polinomios de Lagrange y sus derivadas en xp a partir de los nodos x.
    N = len(x) - 1
    Lagrange = zeros((N+1, N+1, N+1))

### j indica cada uno de los distintos polinomios de Lagrange
    for j in range(N+1):
### El polinomio de Lagrange j de orden 0 es igual a 1
        Lagrange[0, j, 0] = 1
        n = 1
        for i in range(N+1):
            if i != j:
### El orden del polinomio aumenta solo cuando se considera el nodo x[i] (i != j)
                for k in range(N, -1, -1): # k es el orden de derivación
                    Lagrange[k, j, n] = (Lagrange[k, j, n-1] * (xp - x[i]) + k * Lagrange[k-1, j, n-1]) / (x[j] - x[i])
                n = n + 1

### Devuelve los polinomios de Lagrange asociados al nodo x_j evaluados en x = xp
    return Lagrange[:, :, n-1]


def interpolar_densidad_isa(altitudes_isa, densidades_isa, h_objetivo):
    
### Evalúa el interpolante 1D de Lagrange para la densidad ISA en h_objetivo.
    
    L = Lagrange_polynomials(altitudes_isa, h_objetivo)
    return dot(L[0, :], densidades_isa)


def modelo_atmosferico():
    
### Calcula la densidad del aire en el apogeo (h = 405 m) y genera la curva interpolada.
    
### 3. Atmósfera Estándar (ISA): Altitud (m) y Densidad (kg/m^3)
    altitudes_isa = array([0, 500, 1000, 1500, 2000, 2500])
    densidades_isa = array([1.225, 1.167, 1.112, 1.058, 1.007, 0.957])

### Cálculo de la densidad exactamente en el apogeo
    h_apogeo = 405.0
    rho_apogeo = interpolar_densidad_isa(altitudes_isa, densidades_isa, h_apogeo)

### Interpretación física obligatoria por consola
    print("Densidad del aire en el apogeo (h = 405 m) =", rho_apogeo, "kg/m^3")
    print("Interpretación física: A 405 m de altitud la densidad del aire disminuye ligeramente respecto al nivel del mar (1.225 kg/m^3), reduciendo la resistencia aerodinámica en el apogeo.")

### Puntos del polinomio interpolante para representar la curva
    M = 100
    hm = linspace(0, 2500, M)
    rhom = zeros(M)
    for i in range(M):
        rhom[i] = interpolar_densidad_isa(altitudes_isa, densidades_isa, hm[i])

    return altitudes_isa, densidades_isa, hm, rhom, h_apogeo, rho_apogeo
    
### Procedimientos de validación y prueba
def test_apartado3():
    altitudes_isa, densidades_isa, hm, rhom, h_apogeo, rho_apogeo = modelo_atmosferico()

 ### Validación en un nodo conocido (h = 500 m -> rho = 1.167 kg/m^3)
    rho_test = interpolar_densidad_isa(altitudes_isa, densidades_isa, 500.0)
    print("Test validación en nodo h = 500 m (error) =", abs(rho_test - 1.167))

### Representación gráfica
    plt.plot(altitudes_isa, densidades_isa, 'o', label='Datos ISA')
    plt.plot(hm, rhom, '-', label='Interpolante de Lagrange')
    plt.plot(h_apogeo, rho_apogeo, 'rs', label='Apogeo (h = 405 m)')
    plt.xlabel('Altitud h (m)')
    plt.ylabel('Densidad (kg/m^3)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_apartado3()

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

def Plot_Apartado4(ax, t, x, y, ax_slider):

    Vx, Vy, V = Radar_kinematics(t, x, y)


    print("\nApartado 4: Cinemática del Radar")
    print(48*"-")
    print(f"{'t (s)':<10s}{'Vx (m/s)':<13s}{'Vy (m/s)':<13s}{'V (m/s)':<13s}")
    print(48*"-")
    
    for i in range(len(t)):
        print(f"{t[i]:<10.2f}{Vx[i]:<13.2f}{Vy[i]:<13.2f}{V[i]:<13.2f}")
    print("="*54)
    print("Velocidad inicial =", round(V[0], 2), "m/s")
    print("Velocidad en apogeo =", round(V[9], 2), "m/s")
    print("Velocidad final =", round(V[-1], 2), "m/s")
    print("Sentido físico: Vx disminuye por la resistencia del aire y Vy es 0 en el apogeo (t=4.5 s).\n")

### Gráfica y resultados 
def Plot_Apartado4(ax, t, x, y, ax_slider):

    Vx, Vy, V = Radar_kinematics(t, x, y)

    print("\nApartado 4: Cinemática del Radar")
    print(48*"-")
    print(f"{'t (s)':<10s}{'Vx (m/s)':<13s}{'Vy (m/s)':<13s}{'V (m/s)':<13s}")
    print(48*"-")
    
    for i in range(len(t)):
        print(f"{t[i]:<10.2f}{Vx[i]:<13.2f}{Vy[i]:<13.2f}{V[i]:<13.2f}")
    print("="*54)
    print("Velocidad inicial =", round(V[0], 2), "m/s")
    print("Velocidad en apogeo =", round(V[9], 2), "m/s")
    print("Velocidad final =", round(V[-1], 2), "m/s")
    print("Sentido físico: Vx disminuye por la resistencia del aire y Vy es 0 en el apogeo (t=4.5 s).\n")
    
### Gráfica inicial
    ax.clear()
    linea_Vx, = ax.plot(t, Vx, ".-", label="Vx")
    linea_Vy, = ax.plot(t, Vy, ".-", label="Vy")
    linea_V,  = ax.plot(t, V,  ".-", label="V")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Velocidad (m/s)")
    ax.set_title("Apartado 4: Cinemática del Radar")
    ax.legend()
    ax.grid()

    texto = f"V0 = {V[0]:.2f} m/s | V_apogeo = {V[9]:.2f} m/s | Vf = {V[-1]:.2f} m/s\nSentido físico: Vx baja por el drag y Vy = 0 en el apogeo."
    caja_texto = ax.text(0.05, 0.05, texto, transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8))

### Slider de perturbación (para modificar el error de lectura del radar (epsilon en metros))
    ax_slider.clear()
    slider_eps = Slider(ax_slider, "Error radar (m) ", 0.0, 3.0, valinit=0.0, valstep=0.2)

### Recálculo y actualización de las curvas en tiempo real
    def update(val):
        eps = slider_eps.val
        perturbacion = eps * sin(3.0 * t)
        Vx_new, Vy_new, V_new = Radar_kinematics(t, x + perturbacion, y + perturbacion)

        linea_Vx.set_ydata(Vx_new)
        linea_Vy.set_ydata(Vy_new)
        linea_V.set_ydata(V_new)

        caja_texto.set_text(
            f"V0 = {V_new[0]:.2f} m/s | V_apogeo = {V_new[9]:.2f} m/s | Vf = {V_new[-1]:.2f} m/s (eps = {eps:.1f} m)\n"
            f"Sentido físico: Pequeños errores en la posición amplifican oscilaciones al derivar."
        )
        ax.figure.canvas.draw_idle()

    slider_eps.on_changed(update)

    return slider_eps

if __name__ == "__main__":

    t_tel = linspace(0, 8, 17)
    x_tel = array([0, 48, 93, 137, 178, 218, 256, 292, 326, 359, 390, 419, 448, 474, 500, 524, 547])
    y_tel = array([0, 85, 160, 225, 280, 325, 360, 385, 400, 405, 400, 385, 360, 325, 280, 225, 160])

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.22)
    ax_slider = plt.axes([0.22, 0.06, 0.63, 0.04])

    slider = Plot_Apartado4(ax, t_tel, x_tel, y_tel, ax_slider)
    plt.show()

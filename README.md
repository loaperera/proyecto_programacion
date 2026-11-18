# proyecto_programacion

##### APARTADO 2 ####
from numpy import array, polyfit, polyval, linspace, diag, transpose, matmul
from numpy.linalg import svd, solve  
import matplotlib.pyplot as plt

# Método para ajuste lineal por numpy
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

# Método para ajuste lineal manual
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

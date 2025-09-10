import numpy as np

def ecuacion_onda(L, T, c):

    n_x = L*100
    d_x = L/n_x
    CFL = 0.98
    n_t = int(T*c/(CFL*d_x))
    d_t = T/n_t

    u_0_t = 0
    u_L_t = 0

    time_vector = np.linspace(0, T, n_t)  # Vector de tiempo
    x_vector = np.linspace(0, L, n_x)  # Vector de espacio
    u = np.zeros((n_x, n_t))  # Matriz de solución. Cada columna es el desplazamiento de toda la cuerda en un instante de tiempo

    for i in range(n_t):
        for j in range(n_x):

            if i == 0:  # Condición inicial u(x,0) = g(x)
                u[j, i] = g_x(x_vector[j])

            else:
                if j == 0:  # Condiciones de frontera
                    u[j, i] = u_0_t
                elif j == n_x - 1:  # Condiciones de frontera
                    u[j, i] = u_L_t
                else:
                    u[j, i] = 2 * u[j, i - 1] - u[j, i - 2] + (c * d_t / d_x) ** 2 * (
                                u[j + 1, i - 1] - 2 * u[j, i - 1] + u[j - 1, i - 1])

    return u.tolist()

def g_x(x):
    """Desplazamiento inicial u(x,0)."""
    return np.sin(2*np.pi*x)

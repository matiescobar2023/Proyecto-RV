import numpy as np

def ecuacion_onda(L, T, c):
    """Resuelve la ecuación de onda en una cuerda fija en ambos extremos usando el método
    de diferencias finitas explícito.
    Arguments:
    L : float : Longitud de la cuerda (m)
    T : float : Tiempo total de simulación (s)
    c : float : Velocidad de la onda en la cuerda (m/s)

    Returns:
    u : list : Matriz de desplazamientos. Cada columna es el desplazamiento de toda la cuerda en un instante de tiempo.
    """
    # Ecuación de la onda en una cuerda fija en ambos extremos
    # d^2u/dt^2 = c^2 * d^2u/dx^2

    n_x = L*100              # Número de puntos espaciales (más puntos para mayor precisión)
    d_x = L/n_x              # Paso espacial
    CFL = 0.98               # Número de Courant-Friedrichs-Lewy (debe ser <= 1 para estabilidad)
    # El desplazamiento de una perturbación en un intervalo de tiempo d_t no puede ser mayor que el paso espacial d_x
    n_t = int(T*c/(CFL*d_x)) # Número de pasos temporales basado en la condición CFL
    d_t = T/n_t              # Paso temporal

    u_0_t = 0           # Condición de frontera u(0,t) = 0
    u_L_t = 0           # Condición de frontera u(L,t) = 0

    time_vector = np.linspace(0, T, n_t)    # Vector de tiempo
    x_vector    = np.linspace(0, L, n_x)    # Vector de espacio
    u = np.zeros((n_x, n_t))  # Matriz de solución. Cada columna es el desplazamiento de toda la cuerda en un instante de tiempo

    for i in range(n_t):
        for j in range(n_x):

            if i == 0:  # Condición inicial u(x,0) = g(x)
                u[j, i] = g_x(x_vector[j], L)

            elif i == 1:  # primer paso con Taylor + velocidad inicial
                if j == 0:
                    u[j, i] = u_0_t
                elif j == n_x - 1:
                    u[j, i] = u_L_t
                else:
                    u[j, i] = (
                        g_x(x_vector[j])
                        + d_t * v_x(x_vector[j])
                        + 0.5 * (c * d_t / d_x)**2 * (
                            u[j + 1, 0] - 2 * u[j, 0] + u[j - 1, 0]
                        )
                    )
            else:
                if j == 0:          # Condiciones de frontera
                    u[j, i] = u_0_t
                elif j == n_x - 1:  # Condiciones de frontera
                    u[j, i] = u_L_t
                else:
                    u[j, i] = 2 * u[j, i - 1] - u[j, i - 2] + (c * d_t / d_x) ** 2 * (
                                u[j + 1, i - 1] - 2 * u[j, i - 1] + u[j - 1, i - 1])

    return u.tolist()

def g_x(x, L):
    """Desplazamiento inicial u(x,0)."""
    return np.sin(2*np.pi*x/L)

def v_x(x):
    """Velocidad inicial du/dt(x,0)."""
    return 0

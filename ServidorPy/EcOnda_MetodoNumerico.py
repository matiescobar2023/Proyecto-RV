import math
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Contador de computo
f_hz = psutil.cpu_freq().current * 1_000_000


# ∂²u/∂t² = c² * ∂²u/∂x²
# u: desplazamiento vertical
# x: posición horizontal
# t: tiempo
# c: velocidad de la onda en la cuerda
# L: longitud de la cuerda
# f(x): función que describe la forma inicial de la cuerda
# g(x): función que describe la velocidad inicial de la cuerda

# Ecuación diferencial parcial
# cuerda sujeta en sus extremos -> u(0,t) = 0 y u(L,t) = 0
# La cuerda tiene una condición inicial de desplazamiento u(x,0) = g(x)
# La cuerda tiene una condición inicial de velocidad ∂u/∂t(x,0) = f(x)

print("Simulación de la ecuación de la onda - método numérico")

# Parámetros
c = 1  # Velocidad de la onda
L = 1  # Longitud de cuerda
T = 10  # Tiempo total de simulación
n_x = L*100  # Precisión espacial centimétrica
d_x = L/n_x  # Paso espacial
CFL = 0.98  # Condición de estabilidad
n_t = int(T*c/(CFL*d_x))  # Precisión temporal
d_t = T/n_t  # Paso temporal
print(f"Parámetros:\n c = {c} | L = {L} | T = {T} | n_x = {n_x} | CFL = {CFL} | n_t = {n_t}\n")

# Condiciones iniciales
u_0_t = 0
u_L_t = 0
def g_x(x):  # Posicion inicial u(x,0)
    """Desplazamiento inicial u(x,0)."""
    return math.sin(2*math.pi*x)
f_x = 0  # Velocidad inicial ∂u/∂t(x,0)

# Matriz de solucion
time_vector = np.linspace(0, T, n_t)  # Vector de tiempo
x_vector = np.linspace(0, L, n_x)  # Vector de espacio
u = np.zeros((n_x, n_t))  # Matriz de solución. Cada columna es el desplazamiento de toda la cuerda en un instante de tiempo


# Solución numerica

t0 = time.perf_counter_ns()

for i in range(n_t):
    for j in range(n_x):

        if i == 0:  # Condición inicial u(x,0) = g(x)
            u[j, i] = g_x(x_vector[j])

        else:
            if j == 0:  # Condiciones de frontera
                u[j, i] = u_0_t
            elif j == n_x-1:  # Condiciones de frontera
                u[j, i] = u_L_t
            else:
                u[j, i] = 2 * u[j, i - 1] - u[j, i - 2] + (c * d_t / d_x) ** 2 * (u[j + 1, i - 1] - 2 * u[j, i - 1] + u[j - 1, i - 1])

t1 = time.perf_counter_ns()

dt = (t1 - t0) * 1e-9
cycles = int(dt * f_hz)

print(f"Ciclos de cómputo: {cycles}")
print(f"Tiempo de cómputo: {dt:.5f} s")
print(f"tiempo: {d_t: .5f} s")
print("Simulación completada.")


# Visualización

plt.title("Ecuación de la onda - método numérico instante - t = {:.2f}s".format(i * d_t))
plt.grid()
plt.xlabel("Posición (x) [cm]")
plt.ylabel("Desplazamiento (u) [cm]")

step = int(n_t/1000)
for i in range(0, n_t, step):
    plt.clf()  # limpia la figura antes de redibujar
    plt.ylim(-25, 25)
    plt.xlim(0, L * 100)
    plt.plot(u[:, i])
    plt.pause(step*d_t)  # pausa en segundos entre frames
plt.close()
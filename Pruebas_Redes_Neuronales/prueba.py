import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Definir la arquitectura exactamente igual
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.act = nn.Tanh()  # o ReLU, según lo que usaste

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)  # última capa sin activación
        return x

# Crear el modelo con la misma arquitectura
layers = [2, 50, 50, 50, 1] # ejemplo: entradas x, t, C, CI → salida u
model = PINN(layers)

# Cargar los pesos entrenados
model.load_state_dict(torch.load("mired_entrenada.pth"))

# Cambiar a modo evaluación
model.eval()


# ==========================================
#  Generación de animación
# ==========================================
L = 1.0   # largo de la cuerda (1 m)
c = 1.0   # velocidad de onda
T = 10.0   # tiempo de simulación (5 s)

x_test = torch.linspace(0, L, 100).view(-1, 1)
t_test = torch.linspace(0, T, 200).view(-1, 1)
X, Tt = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')

u_pred = []
for tt in t_test:
    xt = torch.cat([x_test, tt * torch.ones_like(x_test)], dim=1)
    u_val = model(xt[:,0:1], xt[:,1:2]).detach().numpy().flatten()  # (100,)
    u_pred.append(u_val)

u_pred = np.array(u_pred).T  # ahora sí shape: (100, 200)


# Animación
fig, ax = plt.subplots()
line, = ax.plot(x_test.numpy(), u_pred[:,0])
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x [m]")
ax.set_ylabel("u(x,t)")

def update(frame):
    line.set_ydata(u_pred[:, frame])
    ax.set_title(f"t = {t_test[frame].item():.2f} s")
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(t_test), interval=50, blit=True)
plt.show()
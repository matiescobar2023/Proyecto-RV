import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
#  Definición de la red neuronal (PINN)
# ==========================================
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = torch.tanh
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)  # Entrada (x,t)
        for i in range(len(self.layers) - 1):
            X = self.activation(self.layers[i](X))
        return self.layers[-1](X)  # Salida u(x,t)

# ==========================================
#  Funciones para gradientes
# ==========================================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def grad2(outputs, inputs):
    g = grad(outputs, inputs)
    return torch.autograd.grad(
        g, inputs,
        grad_outputs=torch.ones_like(g),
        create_graph=True,
        retain_graph=True
    )[0]

# ==========================================
#  Definición del problema
# ==========================================
L = 1.0   # largo de la cuerda (1 m)
c = 1.0   # velocidad de onda
T = 10.0   # tiempo de simulación (5 s)

# Puntos de entrenamiento
N_f = 5000   # puntos internos para PDE
N_b = 200    # condiciones de borde
N_i = 200    # condición inicial

# Malla de entrenamiento
x_f = torch.rand((N_f, 1), requires_grad=True) * L
t_f = torch.rand((N_f, 1), requires_grad=True) * T

# Condiciones de borde: u(0,t)=0, u(L,t)=0
t_b = torch.rand((N_b, 1)) * T
x_b0 = torch.zeros_like(t_b)
x_bL = L * torch.ones_like(t_b)

# Condición inicial: u(x,0)=sin(2πx)
x_i = torch.rand((N_i, 1)) * L
t_i = torch.zeros_like(x_i)
u_i = torch.sin(2 * np.pi * x_i)

# ==========================================
#  Definición del modelo PINN
# ==========================================
layers = [2, 50, 50, 50, 1]
pinn = PINN(layers)

# ==========================================
#  Definición de la función de pérdida
# ==========================================
def pinn_loss():
    # PDE: u_tt = c^2 u_xx
    u_f = pinn(x_f, t_f)
    u_t = grad(u_f, t_f)
    u_tt = grad(u_t, t_f)
    u_x = grad(u_f, x_f)
    u_xx = grad(u_x, x_f)

    f = u_tt - c**2 * u_xx
    loss_pde = torch.mean(f**2)

    # Condiciones de borde
    u_b0 = pinn(x_b0, t_b)
    u_bL = pinn(x_bL, t_b)
    loss_bc = torch.mean(u_b0**2) + torch.mean(u_bL**2)

    # Condición inicial
    u_pred_i = pinn(x_i, t_i)
    loss_ic = torch.mean((u_pred_i - u_i)**2)

    return loss_pde + loss_bc + loss_ic, loss_pde, loss_bc, loss_ic

# ==========================================
#  Entrenamiento
# ==========================================
optimizer = optim.Adam(pinn.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

for epoch in range(2000):
    optimizer.zero_grad()
    loss, lpde, lbc, lic = pinn_loss()
    loss.backward(retain_graph=True)   # 🔑 importante
    optimizer.step()
    scheduler.step() 

    if epoch % 100 == 0:
        print(f"[Adam {epoch:5d}] loss={loss.item():.4e} | pde={lpde.item():.3e}, bc={lbc.item():.3e}, ic={lic.item():.3e}")
    if epoch % 500 == 0:
        print(f"[Iter {epoch}] LR = {scheduler.get_last_lr()[0]:.2e}, loss={loss.item():.3e}")


torch.save(pinn.state_dict(), "mired_entrenada.pth")
# ==========================================
#  Generación de animación
# ==========================================
x_test = torch.linspace(0, L, 100).view(-1, 1)
t_test = torch.linspace(0, T, 200).view(-1, 1)
X, Tt = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')

u_pred = []
for tt in t_test:
    xt = torch.cat([x_test, tt * torch.ones_like(x_test)], dim=1)
    u_val = pinn(xt[:,0:1], xt[:,1:2]).detach().numpy().flatten()  # (100,)
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

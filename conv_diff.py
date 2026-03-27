#微分方程数值解：对流扩散方程的差分法求解
import numpy as np
import matplotlib.pyplot as plt

# 定义方程
# 考虑形如u_t-au_x+\sigmau_xx=0的扩散方程，其中a为对流项，sigma为扩散项
def diffusion(u, a, sigma, dx, dt):
    u_new = u.copy()
    u_new[0:1] = u[1:-1] + dt * (a * (u[2:] - u[:-2]) / (2 * dx) + sigma * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2)
    return u_new

# 定义初始条件,u(x,0)=u_0(x),u(0,t)=u(1,t)=g(t)
def initial_condition(x, t):
    u = np.zeros_like(x)
    u[np.abs(x) < 0.5] = 1
    return u


# ==================== 三种差分格式求解器 ====================

class ConvectionDiffusionSolver:
    def __init__(self, L, N, T, dt, a, sigma):
        """
        L: 空间长度
        N: 空间离散点数
        T: 时间长度
        dt: 时间步长
        a: 对流项
        sigma: 扩散项
        h: 空间步长
        x: 空间网格点
        M: 时间步数

        """
        self.L = L
        self.N = N
        self.T = T
        self.dt = dt
        self.a = a
        self.sigma = sigma
        self.h = L / (N - 1)
        self.x = np.linspace(0, L, N)
        self.M = int(T / dt) + 1

    def method1_explicit_center(self, u0):
        """方法1: 显式中心差分格式"""
        """"U_t-au_x+\sigmau_xx=0的显式中心差分格式"""
        u = np.zeros((self.M, self.N))
        u[0] = u0
        r1 = self.a * self.dt / (2 * self.h)
        r2 = self.sigma * self.dt / (self.h ** 2)

        for m in range(self.M - 1):
            for j in range(1, self.N - 1):
                u[m + 1, j] = u[m, j] + r1 * (u[m, j + 1] - u[m, j - 1]) \
                              - r2 * (u[m, j + 1] - 2 * u[m, j] + u[m, j - 1])
            u[m + 1, 0] = u[m + 1, 1]
            u[m + 1, -1] = u[m + 1, -2]
        return u

    def method2_explicit_upwind(self, u0):
        """方法2: 显式迎风格式"""
        u = np.zeros((self.M, self.N))
        u[0] = u0
        r1 = self.a * self.dt / (2 * self.h)
        r2 = self.sigma * self.dt / (self.h ** 2)

        for m in range(self.M - 1):
            for j in range(1, self.N - 1):
                u[m + 1, j] = u[m, j] + r1 * (u[m, j + 1] - u[m, j]) \
                              - r2 * (u[m, j + 1] - 2 * u[m, j] + u[m, j - 1])
            u[m + 1, 0] = u[m + 1, 1]
            u[m + 1, -1] = u[m + 1, -2]
        return u

    def method3_crank_nicolson(self, u0):
        """方法3: Crank-Nicolson格式"""
        u = np.zeros((self.M, self.N))
        u[0] = u0
        r1 = self.a * self.dt / (4 * self.h)
        r2 = self.sigma * self.dt / (self.h ** 2)

        for m in range(self.M - 1):
            N = self.N - 2
            diag = np.ones(N) * (1 + r2)
            upper = np.zeros(N - 1)
            lower = np.zeros(N - 1)

            for j in range(N - 1):
                upper[j] = -r1 - r2 / 2
                lower[j] = r1 - r2 / 2

            b = np.zeros(N)
            for j in range(N):
                jp = j + 1
                b[j] = u[m, jp] + r2 / 2 * (u[m, jp + 1] - 2 * u[m, jp] + u[m, jp - 1])

            u_new = self.thomas_algorithm(lower, diag, upper, b)
            u[m + 1, 1:-1] = u_new
            u[m + 1, 0] = u[m + 1, 1]
            u[m + 1, -1] = u[m + 1, -2]
        return u

    def thomas_algorithm(self, lower, diag, upper, b):
        n = len(b)
        a = lower.copy()
        c = upper.copy()
        d = b.copy()
        for i in range(1, n):
            m = a[i - 1] / diag[i - 1]
            diag[i] = diag[i] - m * c[i - 1]
            d[i] = d[i] - m * d[i - 1]
        x = np.zeros(n)
        x[-1] = d[-1] / diag[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / diag[i]
        return x


if __name__ == '__main__':
    L, N, T, dt, a, sigma = 1.0, 101, 0.5, 0.001, 0.5, 0.01
    solver = ConvectionDiffusionSolver(L, N, T, dt, a, sigma)
    x = solver.x
    u0 = np.exp(-((x - 0.5) ** 2) / 0.01)

    u1 = solver.method1_explicit_center(u0)
    u2 = solver.method2_explicit_upwind(u0)
    u3 = solver.method3_crank_nicolson(u0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, u, name in zip(axes, [u1, u2, u3],
                           ['Center', 'Upwind', 'Crank-Nicolson']):
        for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            idx = int(t / T * (u.shape[0] - 1))
            ax.plot(x, u[idx], label=f't={t}')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
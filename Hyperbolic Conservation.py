import numpy as np
import matplotlib.pyplot as plt
#考虑一个方程u_t-1/2(f(u))_x=0,不妨考虑f(u)=u^2，这就是无黏度情况下的Burgers方程
#他会生成臭名昭著的“间断解”，考虑一个初值情况u_0(x)=0(x<0)1(x>=1)
def u_initial(x):
    if x<0:
        return 0
    else:
        return 1
#真解实际上可以由特征线法求解u(t,x)=0(x<t)1(x>=t)：
def u_exact(t,x):
    if x<t:
        return 0
    else:
        return 1

#接下来，使用守恒格式来进行求解
class HyperbolicConservationSolver:
    """
    L:空间长度
    T:时间长度
    dt:时间步长
    dx:空间步长
    x:空间坐标
    N:网格点数
    M:时间步数
    """
    def __init__(self,L,N,T,dt):
        self.L=L
        self.N=N
        self.T=T
        self.dt=dt
        self.dx=L/(N-1)
        self.x=np.linspace(0,L,N)
        self.M=int(T/dt)+1

    def method1_explicit(self,u0):
        u=np.zeros((self.M,self.N))
        u[0]=u0
        for m in range(self.M-1):
            for j in range(1,self.N-1):
                u[m+1,j]=u[m,j]-self.dt/(2*self.dx)*(u[m,j+1]**2-u[m,j-1]**2)
            u[m+1,0]=u[m+1,1]
            u[m+1,-1]=u[m+1,-2]
        return u

if __name__ == "__main__":
    # 参数设置
    L = 2.0      # 空间长度
    N = 201      # 空间网格点数
    T = 1.0      # 时间总长
    dt = 0.001   # 时间步长

    solver = HyperbolicConservationSolver(L, N, T, dt)
    x = solver.x

    # 初始条件：阶梯函数
    u0 = np.where(x < 0, 0.0, 1.0)

    # 计算数值解
    u = solver.method1_explicit(u0)

    # 计算真解
    u_exact = np.zeros_like(u)
    for m in range(u.shape[0]):
        t = m * dt
        u_exact[m] = np.where(x < t, -1.0, 1.0)

    # 绘图比较
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 数值解
    times = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for t in times:
        idx = int(t / T * (u.shape[0] - 1))
        axes[0].plot(x, u[idx], label=f't = {t}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].set_title('Numerical Solution')
    axes[0].legend()
    axes[0].grid(True)

    # 真解
    for t in times:
        idx = int(t / T * (u_exact.shape[0] - 1))
        axes[1].plot(x, u_exact[idx], label=f't = {t}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].set_title('Exact Solution')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
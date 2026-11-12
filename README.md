# MyPDESolver

用 Python 做偏微分方程数值求解的学习与复现仓库，重点是随机特征方法（Random Feature Methods, RFM），并与若干经典方法和相关变体对照。

代码实现有 AI 辅助。本仓库目前为私有，用于个人实验记录，不是正式发布的软件包。

## 内容

- `Classical/`：经典数值方法，包括二维 Poisson 的有限差分（`FDM_Possion2d.ipynb`）和有限元（`FEM_Possion.ipynb`）。
- `RFM(PINNs)/`：RFM 及相关实验，包括二维 Poisson（`RFM_Possion2d.ipynb`）、热方程、Burgers 方程等。
- `RFM-DPGM(Deep Galerkin)/`：Deep Petrov–Galerkin 相关实验。
- `RFM-WAN/`：对抗训练相关实验。
- 根目录还有对流扩散（`conv_diff.py`）和双曲守恒律（`Hyperbolic Conservation.py`）脚本。

## RFM 二维 Poisson 复现

在已有代码基础上复现二维 Poisson 算例，并调整了参数与算例。在所测算例上，点态绝对误差约为 $10^{-11}$。实验细节以对应 notebook 为准。

## 已有工作

根据 RFM 阅读笔记和仓库里已经跑过的代码，把目前对得上的工作量记在这里。材料中的 Table 2、Fig. 5、Fig. 6 是文献截图。笔记里的“谱精度”“创新之处”只是原文转录，不作为已核实结论。

阅读上，随机特征方法的做法是：从某个分布采样参数，把输入映射成有限维特征，再在特征空间上用线性模型学习。对应到 PDE，解写成一组参数随机给定、之后固定的单层特征的线性组合，只求组合系数。二维 Poisson 的笔记进一步使用单位分解，把区域分成若干子域，每个子域一组随机特征，用内部和边界配点的残差组成最小二乘。随机参数可以取均匀分布，也可以取更集中的正态分布；单位分解可以用指示函数，也可以用三角函数把子域交界接连续。笔记把所记的那次复现写成 QR 求解，并给出 $L_2$ 误差约 $10^{-11}$ 的误差分布。

代码上，`RFM(PINNs)/RFM_Possion2d.ipynb` 保存了一次可核对的运行：区域分成 `Nx=Ny=2` 块，每块 `M=1600` 个随机特征，配点 `Qx=Qy=40`，线性系统调用 `scipy.linalg.lstsq`（同一笔记本另有 Moore–Penrose 伪逆）。该次输出为点态绝对误差最大值约 $5.59\times 10^{-11}$，打印的 $L_2$ 约 $1.07\times 10^{-11}$。材料里的彩色误差图是这次复现的原图，色标在 $10^{-11}$ 量级，不是论文截图。

同一仓库里还能对上的实验有：`RFM(PINNs)/` 中的热方程、Burgers 方程、局部 ELM，以及另一份二维 Poisson RFM（`RFMinPDE.ipynb`）；`RFM-WAN/Possion_20d_Adversial.ipynb` 中的二十维 Poisson 对抗训练（笔记本留有训练损失）；根目录的对流扩散差分（`conv_diff.py`）和无黏 Burgers 型双曲守恒律（`Hyperbolic Conservation.py`）。`RFM-DPGM(Deep Galerkin)/` 里 Deep Petrov–Galerkin 目前主要是提纲，混合格式笔记本是空的。笔记另记监督学习部分的复现已经做完，对应代码不在本仓库。PDE 实验只跑过一部分，数值以各 notebook 里实际保存的输出为准。

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

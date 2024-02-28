<font size=5>

## 矩阵 Matrix

### 各类型矩阵 Type of Matrix
- 实对称矩阵
$$A^T=A$$
- 实反对称矩阵
$$A^T=-A$$
- 厄尔米特矩阵
$$A^H=A$$
  - $H$ 为 **<font color=gold>共轭转置</font>**
- 反厄尔米特矩阵
$$A^H=-A$$
- 正交矩阵
$$A^TA=AA^T=I$$
    - <font color=gold>$\mathbf{A^T=A^{-1}}$</font>
- 酉矩阵
$$A^HA=AA^H=I$$
    - <font color=gold>$\mathbf{A^H=A^{-1}}$</font>
- 正规矩阵
$$A^HA=AA^H$$
    - 实正规矩阵 <font color=gold>$\mathbf{A^TA=AA^T}$</font>
- 幂等矩阵
$$A^k=A$$
- 正定矩阵
$$\textit{If: } x^TAx>0\quad\textit{Then: A}\;是正定矩阵$$
- 奇异矩阵
$$|A|=0$$

---

### 向量微分
- $\color{chartreuse}\nabla\quad Nabla算子:$
  $
  \text{Def: } \\
  \nabla_{x}f(x_1,\cdots,x_p)=\Big(\cfrac{\partial}{\partial x_1}f,\cdots,\cfrac{\partial}{\partial x_p}\Big)
  \\\quad
  $

- $\color{chartreuse}y=f(x_1,\cdots,x_p)\implies y=X^T\beta$
$$
\nabla_X y=\Big(\cfrac{\partial y}{\partial x_1},\cdots,\cfrac{\partial y}{\partial x_p}\Big)=(\beta_1,\cdots,\beta_p)
\\\quad
$$

- $\color{chartreuse}(y_1,\cdots,y_p)=f(x_1,\cdots,x_p)\implies Y=WX$
$$
\begin{aligned}
 \nabla_X Y&=
    \begin{bmatrix}
        \cfrac{\partial y_1}{\partial x_1} & \cdots & \cfrac{\partial y_1}{\partial x_p}\\
        \vdots& &\vdots\\
        \cfrac{\partial y_p}{\partial x_1} & \cdots & \cfrac{\partial y_p}{\partial x_p}
    \end{bmatrix}_{p\times p}\implies\textcolor{gold}{Jacobian\;Matrix}\\
 &=\begin{bmatrix}
        w_{11} & \cdots & w_{1p}\\
        \vdots& &\vdots\\
        w_{p1} & \cdots & w_{pp}
    \end{bmatrix}_{p\times p}=W
\end{aligned}
\\\quad
$$
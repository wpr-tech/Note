---
print_background: true
---
<font size=5>



### <font color=orange>$\scriptsize ANOVA$ $\scriptsize 方差分析模型$  </font>
- 处理思路：将数据波动分解为<font color=orangered>随机误差</font>和<font color=orangered>不同因素(Factor)的作用</font>
- 线性化数学模型：
  - 单因素方差分析
    $$y_{ij}=\mu+a_i+\varepsilon_i$$
  - 多因素方差分析
    $$y_{ij}=\mu+a_i+b_i+c_i+\varepsilon_i$$
  - 带有交互作用的方差分析
    $$y_{ij}=\mu+a_i+b_i+c_i+\gamma_{ab}+\gamma_{bc}+\varepsilon_i$$
  - 其中，$\mu$ 为所有样本的总均值(不考虑因素差异)；$a_i,\ b_i,\ c_i$ 为不同因素水平变化对总均值的影响；$\gamma_{ab},\ \gamma_{bc}$ 为两因素间的交互效应对总均值带来的影响；$\varepsilon_i$ 为随机误差；$y_{ij}$ 为样本值
  - 值得注意的是模型中$a_i,\ b_i,\ c_i,\ \gamma_{ab},\ \gamma_{bc}$ 其真实形态是<font color=Aqua>$a_i=\sum_{k}\beta_k x_{ik},\ \gamma_{ab}=\sum_{t}\beta_t x_{it}$。其中，$x_{ik},\ x_{it}$ 都为dummy变量(将分类变量转化为one-hot形式), $k=水平数,\ t=因素1的水平数\times 因素2的水平数$</font>
- 固定因素与随机因素：
  - 固定因素：
    该因素的水平都出现了，无需推广。一般固定因素的水平都是<font color=orangered>有限</font>且<font color=orangered>数量较少</font>的
  - 随机因素：
    该因素的水平只出现一部分，需推广至全部水平。一般随机因素的水平都是<font color=orangered>无限</font>或<font color=orangered>数量较多</font>的
- 求解方法：对模型进行回归并对回归系数进行$\small F-test$(此过程等效于方差分析)。

---

### <font color=orange>$\scriptsize ANCOVA\ 协方差分析模型$  </font>
- 在ANOVA当中，自变量都是dummy的。当同时也会存在一些<font color=orangered>与因变量有关的定量连续型变量</font>。这些变量被称为协变量 $Z$，在分析过程需要被固定，因此引入协方差分析。
- 在使用协方差分析模型前要对于自变量 $X$ 与协变量 $Z$ 进行<font color=orangered>平行性检验</font>。即：$Cov(X,Z)=0$
- 线性化数学模型：
  - 单因素协方差分析
    $$y_{ij}=\mu+a_i+\beta(x_{ij}-\mu_x)+\varepsilon_i$$
  - 多因素协方差分析
    $$y_{ij}=\mu+a_i+b_i+c_i+\beta(x_{ij}-\mu_x)+\varepsilon_i$$
  - 带有交互作用的协方差分析
    $$y_{ij}=\mu+a_i+b_i+c_i+\gamma_{ab}+\gamma_{bc}+\beta(x_{ij}-\mu_x)+\varepsilon_i$$ 
- 显然，协方差分析，本质上就是加入了非dummy自变量(协变量 $Z$),经过移项后与$ANOVA\ (1)$ 和 $OLS\ (2)$等效 $$y_{ij}-\beta(x_{ij}-\mu_x)=\mu+a_i+\varepsilon_i\tag{1}$$  $$y_{ij}-a_i=\mu+\beta(x_{ij}-\mu_x)+\varepsilon_i\tag{2}$$
- 求解方法：
  $$\begin{pmatrix}\beta\\\gamma\end{pmatrix}=\Big(\begin{pmatrix}X&Z\end{pmatrix}^T\begin{pmatrix}X&Z\end{pmatrix}\Big)^{-1}\begin{pmatrix}X&Z\end{pmatrix}^TY\\$$
  对模型进行回归并对回归系数进行$\small F-test$

---

### <font color=orange>$\scriptsize LMM\ 线性混合效应模型$  </font>
- 在实际实验中，我们常会遇到<font color=Aqua>多个受试者在多个不同时间的研究数据</font>，并使用他们来研究因变量与自变量之间的关系。
- 值得注意的是，此种数据在收集过程中会存在<font color=orangered>时间点无法对齐</font>和<font color=orangered>不同受试个体收集的数据量不同</font>的问题。这导致我们不能使用多元时间序列分析的模型；同时由于任务目的需要，此类问题更适合回归模型，因此我们引入线性混合效应模型。
- 固定效应 & 随机效应
  $$\begin{cases}固定效应:解释因变量的连续型和分类型自变量\\随机效应:描述个体之间的差异的变量\end{cases}$$
- 线性化数学模型：
  组内模型（单个受试的数据集）
  $$y_i=X_i\beta+Z_i\bm{b}_i+\varepsilon_i$$ $$\bm{b}_i\backsim N_q(0,\sigma^2D),\ \varepsilon_i\backsim N_{n_i}(0,\sigma^2R_i)$$ 
  整体模型：
  $$y=X\beta+Z\bm{b}+\varepsilon$$ $$\bm{b}_i\backsim N(0,\tilde{D}),\ \varepsilon_i\backsim N(0,\sigma^2R)$$ $$\tilde{D}=diag(D,D,\cdots,D),\ R=diag(R_1,R_2,\cdots,R_N)$$
- 求解方法：
  使用$y$ 的边缘分布并利用极大似然法 $ML$ 或限制极大似然法 $REML$ 求解。
- <font color=orangered>$PS$</font> :
  $X$ 中可以包括交叉效应。
  $Z$ 中一般为常数项和 $X$ 中的变量。
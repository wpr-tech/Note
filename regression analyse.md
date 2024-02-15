---
print_background: true
---
<font size=5>

### <font color=orange>endog & exog</font> 
- endog --> y --> Endogenous(内生变量) **模型内需解释的变量**
- exog ---> x ---> Exogenous(外生变量) **模型无需解释的变量**

### <font color=orange>Gaussian-Markov Theory</font>
 $$if\ E(\varepsilon)=0, Var(\varepsilon)=\sigma^2 I_n,\ \text{E}(X,\varepsilon)=0\implies OLS=BLUE$$

### <font color=orange>$\scriptsize 计量经济学假设$</font>
- <font color=Chartreuse>1. 线性假设：</font>
    $$Y=X\beta+\varepsilon$$
    使用<font color=Aqua>$F-test$</font>检验
- <font color=Chartreuse>2. 零均值假设/严格外生假设：</font>
    $$\text{E}(\varepsilon_i|X）=\text{E}(\varepsilon_i|X_1,\cdots,X_i,\cdots,X_n)=0$$
- <font color=Chartreuse>3. 同方差假设/球形干扰：</font>
    $$Var(\varepsilon)=\sigma^2I_n=\begin{cases}
    Var(\varepsilon_i|X_i)=E(\varepsilon_i^2)=\sigma^2【同方差性】\\
    Cov(\varepsilon_i,\varepsilon_j)=0【序列相关性】
    \end{cases}$$
    - 使用<font color=Aqua>$White-test,\ GQ-test,\ BP-test$</font>检验异方差性，使用<font color=Aqua>$WLS$</font>补救
    - 使用<font color=Aqua>$DW-test,\ BG-test,\ LM-test$</font>检验序列相关性，使用<font color=Aqua>$GLS$</font>补救
- <font color=Chartreuse>4. 无共线性假设/满秩假设：</font>
    $$rank(X)=p$$
    - 使用<font color=Aqua> $VIF,Cov\ Matrix,\ Cond.No$</font>检验多重共线性
- <font color=Chartreuse>5. 正态性假设：</font>
    $$\varepsilon|X\backsim N(0,\sigma^2I_n)$$
    - 使用<font color=Aqua>$JB-test$</font>检验正态性
### <font color=orange>$\scriptsize 检验回归系数显著性$</font>
- 小样本+单约束$\implies T-test$ 
- 渐进相等+大样本+多约束$\implies\begin{cases}LR-test\\Wald-test\\LM-test\end{cases}$
---

### <font color=orange>*$\scriptsize OLS$* </font> <font color=Aqua> $\scriptsize [\varepsilon\backsim N(0,\sigma^2 I)]$</font>
$$Y=X\beta+\varepsilon\implies\min_{\hat{\beta}}\sum_{i=1}^n(y_i-X_i\hat{\beta})^2$$
$$\implies\hat{\beta}=(X^T\cdot X)^{-1}\cdot X^T\cdot Y$$

- <font color=red>**特殊的**：</font>$y=\beta_{0}+\beta_{1}x$

### <font color=orange>*$\scriptsize OLS\  non-linear curve$*</font>
- 是一种特殊形式的***OLS***
$$y = \beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+\beta_{3}\sin(x_1)+\beta_{4}x_{2}^2+beta_{5}x_1x_{2}+\varepsilon$$
- 可将$x_{1}x_{2}$, $\sin(x_1)$ or $x_{2}^2$ 看作新的$x$

### <font color=orange>*$\scriptsize OLS\ with\ dummy\ variables$*</font>
- 是一种特殊形式的***OLS***
$$y = \beta_{0}+\beta_{1}x_{1}+\beta_{2}x_{2}+\varepsilon$$
- 其中$x_{2}$为分类变量
  
---

### <font color=orange>$\scriptsize 衡量指标$</font>
### 1. <font color=Chartreuse>$\scriptsize R^2$</font>: $\scriptsize 可决系数$ 
因变量y的波动有多少比例可以由自变量来解释
$$R^2=1-\frac{SSE}{SST}$$
- $SSE=\sum (\hat{y_{i}}-\bar{y_{i}})^2$ 是残差平方和
- $SST=\sum (y_{i}-\bar{y_{i}})^2$ 是总体标准差
- $Std\ err=\sqrt{SST/n}$ 是标准误
- $SSR=\sum (\hat{y_{i}}-y_{i})^2$ 是拟合残差。
- **$SST=SSR+SSE$**
  
### 2. <font color=Chartreuse>$\scriptsize Adjust\ R^2$</font>: $\scriptsize R^2 的无偏估计$
$$R^2(adj)=1-(1-R^2)\frac{n-1}{n-p-1}=1-\frac{\frac{1}{n-p-1}SSR}{\frac{1}{n-1}SST}$$

### 3. <font color=Chartreuse>$\scriptsize F-statistic$</font>：$\scriptsize F检验，判断线性关系$
- $H_{0}:\beta_{1}=\cdots=\beta_{p}=0$
$$F=\frac{SSR/p}{SSE/(n-p-1)}\backsim F(p, n-p-1)$$

### 4. <font color=Chartreuse>$\scriptsize Log-Likelihood$</font>：$\scriptsize 对数似然值，判断拟合优度$
- 高斯分布模型：
$$\ln(L(\theta|x))=\ln(\displaystyle\prod_{i=1}^n p(y|x;\beta))=\ln(\displaystyle\prod_{i=1}^n\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y_i-X_i\beta)^2}{2\sigma^2}})$$

### 5. <font color=Chartreuse>$\scriptsize Akaike\ Information\ Criterion - AIC$</font>：$\scriptsize 判断拟合优度$
$$AIC  =-2\times\ln(L(\theta|x))+2p$$

### 6. <font color=Chartreuse>$\scriptsize Bayesian\ Information\ Criterion - BIC$</font>：$\scriptsize 判断拟合优度$
- 相比于$AIC$, $BIC$ 可以有效地减少 <font color=orangered>**过拟合** </font>和<font color=orangered>**维度灾难** </font><font color=aqua>（<u>维度越大，效果越差</u>）</font>
$$BIC  =-2\times\ln(L(\theta|x))+p\times\ln(n)$$

### 7. <font color=Chartreuse>$\scriptsize T-test$</font>： $\scriptsize 回归系数检验$
- $H_{0}:\beta_{i}=0$
- $H_{1}:\beta_{i}\neq0$
$$t=\frac{\hat{\beta_{1}}}{s_{\hat{\beta_{1}}}}$$

### 8. <font color=Chartreuse>$\scriptsize Durbin-Watson\ test$</font>: $\scriptsize 检验样本残差间是否AR(1)$
- $\varepsilon_{i}=\rho\varepsilon_{i-1}+v_t$
- $H_{0}:\rho=0$
- $H_{1}:\rho\neq 0$
$$d=\frac{\sum^n_{i=2}(\varepsilon_{i}-\varepsilon_{i})^2}{\sum^n_{i=2}\varepsilon_{i}^2}$$
- $d$ 越接近2越好，$d=1\backsim 3$ 没问题，$d<1$ 否定$H_{0}$

### 9. <font color=Chartreuse>$\scriptsize Jarque-Bera\ test$</font>: $\scriptsize 检验数据是否具有正态性$
- 使用时：$N>30$
- $H_{0}:X\backsim N$
$$J-B=\frac{n}{6}(Skew^2+\frac{(Kurtosis-3)^2}{4})$$

### 10.<font color=Chartreuse>$\scriptsize  Likelihood\ Ratio-test$ </font>：$\scriptsize 只适用线性$
- 作用同$\ Wald-test$, 使用时需分别计算约束与非约束模型的 $\log{L(\beta)}$
- $H_{0}:\hat{\beta}_{1\times q}=\{\hat{\beta}_i,\cdots,\hat{\beta}_j\}_{1\times q}=0, q\leq p$
- $\beta_{constent}=\complement_{\beta_0}(\beta_0\cap\beta_{constent}),\ \beta_{non-constent}=\beta_0$
$$LR=2(\log{L(\beta_{constent})}-\log{L(\beta_{non-constent})})\backsim\chi^2(q)$$
- $q$ 为约束变量个数（下同）

### 11. <font color=Chartreuse>$\scriptsize Wald-test$ </font>：$\scriptsize (适用线性and非线性方程)$
检验使用规模更大的模型是否比规模更小的模型有<u>**更好的拟合度**</u>，若没有，则没有必要选择规模更大的模型。(规模：回归系数个数)
  
- $H_{0}:f(\hat{\beta})=f({\beta}_1,\cdots,\hat{\beta}_p)=0$
$$W = (f(\hat{\beta}))^T[Var(f(\hat{\beta}))]^{-1}(f(\hat{\beta}))\backsim \chi^2(q)$$

- <font color=Aqua>$eg:$ $f(\hat{\beta})=\beta_1\beta_2-\beta_3=0$</font>
- $Var(f(\hat{\beta}))=(\frac{\partial f(\hat{\beta})}{\partial\hat{\beta}})(Var(\hat{\beta}))(\frac{\partial f(\hat{\beta})}{\partial\hat{\beta}})^T$

### 12. <font color=Chartreuse>$\scriptsize Lagrange Multiplier-test$ </font>:$\scriptsize (适用线性and非线性方程)$
- 思路： 若$x_i,x_j$ 可去，则$x_i,x_j$与$y$无关，更不会进入$\varepsilon$中，由$Gaussian-Markov\ Theory$可知，$\varepsilon$与其他 $x_k, k\neq i,j$也无关。
    <font color=orangered>

    **$\text{Step} 1$**:</font> 对 $y=\sum_{k\neq i,j}\beta_{k}x_{k}+\varepsilon$ 进行回归，得到残差序列 $\varepsilon$。
    <font color=orangered>
    
    **$\text{Step} 2$**:</font> 对辅助回归方程$\varepsilon=\sum_{k=0}^p\alpha_kx_k+\varepsilon_u$ 进行回归，并计算可决系数$R_u^2$。
    <font color=orangered>

    **$\text{Step} 3$**:</font> 构建$LM$统计量：$LM=nR_u^2\backsim\chi^2(q)$。

- $LM$统计量还可以检测残差的高阶延后 $AR(p)$
    辅助统计量为：<font color=orangered>$(BG-test)$</font>
    $$\varepsilon=\sum_{k=1}^m\beta_{k}x_k+\sum_{i=1}^p\rho_iu_{t-i}+v_t$$
    $$LM=(n-p)R_u^2\backsim\chi^2(p)$$

### 13. <font color=Chartreuse>$\scriptsize White-test$ </font>：$\scriptsize (检验异方差)$
- $H_{0}:\forall\varepsilon_i\backsim N(0,\sigma^2)\implies同方差$
- 辅助方程：$\varepsilon^2_i=\alpha_0+\alpha_1x_{1i}+\alpha_2x_{2i}+\alpha_3x^2_{1i}+\alpha_4x^2_{2i}+\alpha_5x_{1i}x_{2i}+v_i$ **<font color=orangered><u>(包含：一次项、二次项&交叉项)**</font></u>
$$W=nR_{W}^2\backsim\chi^2(m)$$
- $m$ 为辅助回归模型回归系数个数

### 14. <font color=Chartreuse>$\scriptsize BP-test$ </font>：$\scriptsize(检验异方差，White-test的特例)$
- $H_{0}:\forall\varepsilon_i\backsim N(0,\sigma^2)\implies同方差$
- 辅助方程：$\varepsilon^2_i=\alpha_0+\alpha_1x_{1i}+\alpha_2x_{2i}+v_i$ **<font color=orangered><u>(包含：一次项)**</font></u>
$$W=nR_{B}^2\backsim\chi^2(m)$$
- $m$ 为辅助回归模型回归系数个数

### 15. <font color=Chartreuse>$\scriptsize GQ-test$ </font>：$\scriptsize (检验异方差,使用F-test)$
<font color=orangered>

$\text{Step} 1$</font>：将模型观测值按照解释变量大小升序排列 
<font color=orangered>
    
$\text{Step} 2$</font>:将排序后样本中间删掉 $c$ 个样本 **<font color=orangered>(一般取样本数的四分之一)</font>**，将余下的样本平均分为两部分，每部分有$\frac{n-c}{2}$个观测值。 
<font color=orangered>

$\text{Step} 3$</font>:对两个样本进行回归，得到残差平方和$SSE_1, SSE_2$ 构建统计量
$$F=\frac{\cfrac{SSE_1}{\frac{n-c}{2}-1}}{\cfrac{SSE_2}{\frac{n-c}{2}-1}}\backsim F(\frac{n-c}{2}-k-1,\ \frac{n-c}{2}-k-1)$$

### 15. <font color=Chartreuse>$\scriptsize Gleiser-test\ and\ Park-test$ </font>：$\scriptsize (检验异方差,并寻找出异方差形式\ \sigma_i=f(X_i))$
<font color=orangered>

$\text{Step} 1$</font>：对模型进行 $\small OLS$ 回归，得到残差序列 $\{\hat{\varepsilon}_i\}_{i=1}^n$
<font color=orangered>

$\text{Step} 2$</font>：对于辅助回归模型进行回归
$$\begin{aligned}Gleiser: \ln{|\hat{\varepsilon_i}|}&=\ln{\sigma^2}+\alpha\ln{X_i}+v_i\\Park: \ln{(\hat{\varepsilon_i}^2)}&=\ln{\sigma^2}+\alpha\ln{X_i}+v_i\end{aligned}
$$
其中，$\ln{\sigma^2}$为辅助回归模型的常数(截距)项
<font color=orangered>

$\text{Step} 3$</font>：对于辅助回归模型的回归系数 $\alpha$ 进行 $T-test$ , 检验回归系数显著性。若显著，则存在异方差性并确定异方差形式 $f(X_i)$

- 检验的辅助回归模型与异方差形式 $f(X_i)$的关系为：
  $$\sigma_i=f(X_i)=\sigma^2X_i^{\alpha}e^{v_i}\overset{\mathit{ln}}{\implies}\ln{\sigma^2}+\alpha\ln{X_i}+v_i$$


### <font color=OrangeRed>$\scriptsize PS$</font>: 
- <font color=Chartreuse>$Df\ Residuals$</font>: 残差自由度 $(n-p-1)$
- <font color=Chartreuse>$Df\ Model$</font>: 模型参数个数 $(p)$
- <font color=Chartreuse>$Skewness->Skew$</font>: 偏度
$$Skew=E[(\frac{(Mean-Median)}{Std})^3]=\frac{1}{n}\sum^{n}_{i=1}[(\frac{X_i-\mu}{\sigma})^3]$$
- <font color=Chartreuse>$Kurtosis$</font>: 峰度
$$Kurtosis=E[(\frac{(Mean-Median)}{Std})^4]=\frac{1}{n}\sum^{n}_{i=1}[(\frac{X_i-\mu}{\sigma})^4]$$
- <font color=Chartreuse>$P>|t|$</font>: $P-value$
- <font color=Chartreuse>$[0.025\ 0.975]$</font>: 系数的95%置信区间
- <font color=Chartreuse>$Cond. No.$</font>: $\ if\ Cond. No.<100$，共线性程度小,$\ if\ 100<Cond. No.<1000$，共线性程度较大,$\ if\ Cond. No.>1000$，共线性程度严重
  
---

### <font color=orange>$\scriptsize GLS$ $\scriptsize 广义最小二乘法$  </font><font color=Aqua>$\scriptsize[\varepsilon\backsim N(0,\sigma^2\Sigma),\ \Sigma 可逆]$</font>

$$\tilde{Y}=\tilde{X}\beta+\varepsilon\implies\min_{\hat{\beta}} \frac{1}{2p}(Y-X\hat{\beta})^T\Sigma^{-1}(Y-X\hat{\beta})$$
$$\implies\hat{\beta}=(X^T\Sigma^{-1}X)^{-1}\cdot X^T\Sigma^{-1} Y=(\tilde{X}^T\cdot \tilde{X})^{-1}\cdot \tilde{X}^T\cdot \tilde{Y}$$
- $\Sigma = C^TC$
- $\tilde{X}=CX$, $\tilde{Y}=CY$

##### <font color=orangered>误差项遵循</font> $\scriptsize AR(1)$
$$\varepsilon_{i}=\beta_{0}+\rho\varepsilon_{i-1}+\eta_{i},\ \eta_{i}\backsim N(0,\Sigma)$$
- 可使用GLSAR等带有AR的 **<font color=orangered>滞后回归模型</font>**
- $\small FGLS$ (可行广义最小二乘法)：因一般残差的协方差阵不一致，需使用样本残差协方差阵进行一致估计后得出残差的协方差, 再使用$\small GLS$

---

### <font color=orange>$\scriptsize Quantile\  Regression$</font>: $\scriptsize分位数回归$
$$Y=X\beta+\varepsilon\implies\min_{\hat{\beta}}\frac{1}{n}(\sum_{i:Y_{i}<X\hat{\beta}}(1-\tau)|y_{i}-X\hat{\beta}|\sum_{i:Y_{i}\geq X\hat{\beta}}\tau|y_{i}-X\hat{\beta}|) $$
- 这里的分位数是 **<font color=orangered>下分位数</font>**，即回归线下包含<font color=Aqua>$\tau\times 100\%$</font>的数据
#### <font color=Chartreuse>分位数</font>
$$\tau=\text{P}(y\leq y_{\tau})=\text{F}_{\tau}(y)$$

#### <font color=Chartreuse>$\scriptsize LAD\ Estimator$</font>: $\scriptsize 最小一乘回归(0.5分位数回归)$
$$\min_{\hat{\beta}}\frac{1}{n}\sum^n_{i=1}|y_{i}-X\hat{\beta}|$$

#### <font color=Chartreuse>分位数相关检验</font>
- 拟合优度检验：
  
  需将解释变量矩阵与系数向量分为两部分
  $$X=(1_{n\times 1},Z_{n\times p})_{n\times p},\ \ \hat{\beta}_{(\tau)}=(\hat{\beta}_{0(\tau)},\hat{\beta}_{2\backsim p(\tau)})$$
  <font color=orangered>$\hat{\beta}_{2\backsim p(\tau)}\neq 0$ 时的残差
  </font>
  <font size=4>
  $$\hat{Q}=\min{\displaystyle\sum_{i:y_i<X\hat{\beta}}(1-\tau)|y_i-\hat{\beta}_{0(\tau)}-Z\hat{\beta}_{2\backsim p(\tau)}|+\displaystyle\sum_{i:y_i\geq X\hat{\beta}}(1-\tau)|y_i-\hat{\beta}_{0(\tau)}-Z\hat{\beta}_{2\backsim p(\tau)}|}$$
  </font>
  
  <font color=orangered>$\hat{\beta}_{2\backsim p(\tau)}=0$ 时的残差
  </font>
  <font size=4>
  $$\tilde{Q}=\min{\displaystyle\sum_{i:y_i<X\hat{\beta}}(1-\tau)|y_i-\hat{\beta}_{0(\tau)}|+\displaystyle\sum_{i:y_i\geq X\hat{\beta}}(1-\tau)|y_i-\hat{\beta}_{0(\tau)}|}$$
  </font>
  <font color=orangered>
  
  **拟合优度：**</font>
  使用分位数回归和使用水平直线回归的拟合残差之差
  
  $$R^*_{(\tau)}=1-\frac{\hat{Q}}{\tilde{Q}}$$


  
- 系列分位数检验
    - 斜率相等检验：检验不同分位数回归下相同特征的回归系数是否相同
      
      $H_{0}:\beta_{i(\tau_1)}=\cdots=\beta_{i(\tau_m)},\ i=1,\cdots,k$, 使用$Wald-test$进行检验，其服从 $\chi^2((k-1)(m-1))$

    - 对称性检验: 检验y的分布是否对称
      
      $H_{0}:\frac{\beta_{0,(\tau_j)}+\beta_{0,(\tau_{m-j+1})}}{2}=\beta_{0,(0.5)}$, 使用$Wald-test$进行检验，其服从 $\chi^2(k(m-1)/2)$

---

### <font color=orange>$\scriptsize RecursiveLS$</font>: $\scriptsize 递归最小二乘回归$
- 该方法与数据不同时间批次出现有关
- 为减少多次回归的计算量，**<font color=orangered>引入参数更新方法</font>**
    **<font color=orangered>已知</font>**：$$\hat{\beta}_{0}=(X_{0}^TX_{0})^{-1}X_{0}^TY=\Sigma_{0}^{-1}X_{0}^TY$$
    $$\implies\Sigma_{0}\hat{\beta}_{0}=X_{0}^TY\tag{1}$$

    在引入新数据$X_{1}$ 后，再次求解就会变为：
    $$\hat{\beta}_{1}=\Big(\begin{bmatrix}X_{0}\\X_{1}\end{bmatrix}^T\begin{bmatrix}X_{0}\\X_{1}\end{bmatrix}\Big)^{-1}\begin{bmatrix}X_{0}\\X_{1}\end{bmatrix}^T\begin{bmatrix}Y_{0}\\Y_{1}\end{bmatrix}=\Sigma_{1}^{-1}\begin{bmatrix}X_{0}\\X_{1}\end{bmatrix}^T\begin{bmatrix}Y_{0}\\Y_{1}\end{bmatrix}$$
    $$\implies\Sigma_{1}\hat{\beta}_{1}=\begin{bmatrix}X_{0}\\X_{1}\end{bmatrix}^T\begin{bmatrix}Y_{0}\\Y_{1}\end{bmatrix}$$

    经过递推可得：
    $$\Sigma_{k}=\Sigma_{k-1}+X_{k}^TX_{k}\tag{2}$$
    由(1)和(2)可推出：
    $$\begin{aligned}\begin{bmatrix}X_{0}\\X_{1}\end{bmatrix}^T\begin{bmatrix}Y_{0}\\Y_{1}\end{bmatrix}&=X_{0}^TY_{0}+X_{1}^TY_{1}\\&=\Sigma_{1}\hat{\beta}_{0}+X_{1}^T(Y_{1}-X_{1}\hat{\beta}_{0})\end{aligned}$$
    可递推出：
    $$\hat{\beta}_{k}=\hat{\beta}_{k-1}+\Sigma_{k}^{-1}X_{k}^T(Y_{k}-X_{k}\hat{\beta}_{k-1}) \tag{3}$$

    我们可以根据两个递推式来更新回归系数，但对矩阵求逆无疑是困难且复杂的。因此我们使用Sherman-Morrison-Woodbury 引理来简化运算：
    $$\mathit{Lemma}:\small Sherman-Morrison-Woodbury\\(A+UV^T)^{-1}=A^{-1}-A^{-1}U(I+V^TA^{-1}U)^{-1}V^TA^{-1}$$

    将(2)代入(3)中，并应用引理将两个递推式转化为：
    $$P_k=\frac{1}{\lambda}(P_{k-1}-P_{k-1}X_k^T(\lambda I+X_kP_{k-1}X_k^T)^{-1}X_kP_{k-1})\\\hat{\beta}_k=\hat{\beta}_{k-1}+P_kX_k^T(Y_k-X_k\hat{\beta}_{k-1})$$

    其中，$P_k=\Sigma_{k}^{-1}$, $\lambda$为遗忘因子。且 $\lambda I+X_kP_{k-1}X_k^T$ 为 **<font color=orangered>一个实数</font>**，取逆=取倒数
- 遗忘因子的加入，解决了旧数据因数量过大而淹没新数据的问题。使用遗忘因子的算法称为 $FFRLS$, 当 $lambda=1$ 时，$FFRLS\implies RecursiveLS$
- $RecursiveLS$ 常用作自适应滤波器
- $\small CUSUM-test$ 是残差图评价方法的拓展，他引入了递归残差的累计残差和。
    </br>可使用 $\small CUSUM检验$ 和 $\small CUSUM平方检验$
    $H_{0}:参数稳定\\H_{1}:参数非稳定$
    **<font color=orangered>判断标准</font>**： 若图中曲线超出设定的置信区间则否定 $H_{0}$
    **<font color=orangered>参数稳定性假设</font>**：
    有两个使用不同数据子集进行回归的回归模型：
    $$Y=X\beta+\varepsilon\\ Y=X\alpha+v$$
    若：$\alpha=\beta$, 则称模型参数稳定。

---

### <font color=orange>$\small Rolling\ Regression$</font>: $\scriptsize 滚动窗口回归$
- 滚动窗口有两种形式：
    $$\begin{cases}
   固定起始值：&给定起始窗口长度，给定最大窗口长
   度，\\&在回归过程中窗口长度逐渐递增\\
   固定窗口值：&给定窗口固定长度\\&在回归过程中窗口长度始终不变
    \end{cases}$$
- 显然这与滑动窗口的两种形式类似，本质上:
    $$\small Rolling\ Regression=Sliding\ Window+OLS$$

---

### <font color=orange>$\scriptsize WLS$ $\scriptsize 加权最小二乘法$  </font><font color=Aqua>$\scriptsize[\varepsilon_i\backsim N(0,\sigma_i^2),异方差性]$</font>
$$Var(\varepsilon)=\begin{bmatrix}\sigma_1^2&0&\cdots&0\\0&\sigma_2^2&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\sigma_n^2\end{bmatrix}\implies w_i=\frac{1}{\sigma_i^2}$$
$$\implies W=\begin{bmatrix}w_1&0&\cdots&0\\0&w_2&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&w_n\end{bmatrix}$$

$$Y=X\beta+\varepsilon\implies\min_{\hat{\beta}} \frac{1}{2p}(Y-X\hat{\beta})^T\Sigma^{-1}(Y-X\hat{\beta})$$
$$\implies\hat{\beta}=(X^TW^{-1}X)^{-1}\cdot X^TW^{-1} Y$$
- $\small FWLS$ (可行加权最小二乘法)：因一般残差的协方差阵不一致，需使用样本残差协方差阵进行一致估计后得出残差的协方差, 再使用$\small WLS$
- 显然$\small WLS$是$\small GLS$的一种特殊形式。因此在异方差和序列相关同时存在时，使用$\small GLS$进行回归。

---



    


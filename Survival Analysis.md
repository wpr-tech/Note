<font size=5>

## Survival Analysis
### <font color=Aqua>$\small Censored\ Data$ \<删失数据\></font>
- 生存分析主要处理一种特殊的时间数据，并且时间数据可能带有部分删失属性
- 盲目的删除 $\text{Censoring Data}$ 会 **<font color=orangered>提高模型的方差</font>**，以至于 **<font color=orangered>降低估计精度</font>**
- 删失数据产生原因：
  $$
  \begin{cases}
  因实验终止而无法观测\implies 恢复较好\\
    \left. 
      \begin{array}{lll}
      因情况变差而自动退出\\
      因前往其他地区治疗而无法参与实验
      \end{array}
    \right   \}\implies恢复可能不理想
  \end{cases}$$
- 删失数据如何带来偏差：
  $$不考虑病人提前退出的原因\implies过高估计存活时间$$

#### <font color=Chartreuse>$\small T's\ pdf\quad f(t)$ \<事件的概率密度函数\></font>
$$f(t)=P(T=t)=p(t)$$
- 量化事件在 $t$ 时间 **<font color=Aqua>时</font>** 发生的概率

#### <font color=Chartreuse>$\small T's\ cdf\quad F(t)$ \<事件的分布函数\></font>
$$F(t)=P(T\leq t)$$

#### <font color=Chartreuse>$\small Survival\ Function\quad S(t)$ \<生存函数\></font>
$$S(t)=1-F(t)=P(T>t)$$
- 量化事件在 $t$ 时间 **<font color=Aqua>之后</font>** 发生的概率
$$
\begin{cases}
Censored\ Data\implies Died &Underestimate\ S(t）\\
Censored\ Data\implies  Alive &Overestimate\ S(t)
\end{cases}
$$
- 重要结论：
$$S'(t)=-F'(t)=-f(t)\leftrightarrow f(t)=-S'(t)$$

#### <font color=Chartreuse>$\small Hazard\ Function\quad\lambda(t)$ \<风险函数\></font>
$$\begin{aligned}
\lambda(t)&=\lim_{h\rightarrow0^+}\underbrace{\cfrac{p(t<T\leq t+h|T>t)}{h}}_\text{拆分条件概率}\\
&=\cfrac{1}{P(T> t)}\lim_{h\rightarrow0^+}\cfrac{P(t<T\leq t+h)}{h}\\
&=\cfrac{f(t)}{S(t)}
\end{aligned}
$$
- 量化事件在 $t$ 时间 **<font color=Gold>之后瞬时发生的可能性</font><font color=orangered>（不是概率）</font>**
- 有一个重要结论：$\lambda(t)=-\cfrac{d}{dt}\mathit{ln}S(t)$
  $$
  \begin{aligned}
  \lambda(t)&=\cfrac{f(t)}{S(t)}=\cfrac{d}{dt}\int\cfrac{f(t)}{S(t)}dt\\
  &=\cfrac{d}{dt}\int\cfrac{1}{S(t)}d\Big(-S(t)\Big)\\
  &=-\cfrac{d}{dt}\mathit{ln}S(t)
  \end{aligned}
  $$

#### <font color=Chartreuse>$\small Cumulative\ Hazard\ Function\quad\Lambda(t)$ \<累积风险函数\></font>
$$\Lambda(t)=\int_0^t\lambda(s)ds$$
- 量化事件在 $t$ 时间点前的风险和

#### <font color=Chartreuse>$\small Mean\  Residual\ Life\quad r(t)$ \<累积风险函数\></font>
$$r(t)=\mathit{E}(T-t|T\geq t)=\cfrac{\int_t^{\infin}S(u)du}{S(t)}$$
- 量化事件在 $t$ 时间 **<font color=Aqua>后</font>** 发生的期望值
- 证明过程：
  $Lemma\ 1:\ E(x)=\displaystyle\int_{-\infty}^{\infty} x\cdot p(x)dxdx=\int_0^{\infty}P(X\geq x)dx$
  $$
  \begin{aligned}
  Proof:\ E(x)&=\displaystyle\int_{-\infty}^{\infty}x\cdot p(x)dx\\
  &=\int_{-\infty}^{\infty}\int_0^xp(y)dydx\\
  &=\int_{-\infty}^{\infty}\int_{x}^{\infty}p(y)dydx=\int_0^{\infty}1-F(x)dx\ \Box
  \end{aligned}
  $$
  </br>$Lemma\ 2:\ E(x|y)=\displaystyle\int_{-\infty}^{\infty} x\cdot p(x|y)dx=\displaystyle\int_{-\infty}^{\infty} x\cdot\cfrac{p(x,y)}{p(y)}dx$
  </br>$Proof:\ 显然$
  $$
  \begin{aligned}
  Proof:&\ Mean\  Residual\ Life\\r(t)&=\mathit{E}(T-t|T>t)=\underbrace{\int_{-\infty}^{\infty}P(T-t>s|T>t)ds}_\text{Lemma 1}\\
  &=\underbrace{\int_{-\infty}^{\infty}\cfrac{P(T-t>s,T>t)}{P(T>t)}ds}_\text{Lemma 2}\\
  &=\int_{-\infty}^{\infty}\cfrac{P(T-t>s)}{P(T>t)}ds=\cfrac{1}{S(t)}\int_{-\infty}^{\infty}P(T-t>s)ds\\
  &=\cfrac{1}{S(t)}\int_{-\infty}^{\infty}P(T>t+s)ds\\&=\cfrac{1}{S(t)}\int_{t}^{\infty}P(T>t+s)d(t+s)\\
  &\underset{u=t+s}{\implies}\cfrac{1}{S(t)}\int_{t}^{\infty}P(T>u)d(u)=\cfrac{\int_t^{\infin}S(u)du}{S(t)}\ \Box
  \end{aligned}
  $$
#### <font color=Chartreuse>$\small Relationship$ \<函数之间的关系\></font>
![](Survival_Func.png)  

---

### <font color=Aqua>$\small Likelihood$ \<删失数据的似然\></font>
#### <font color=Chartreuse>$\small Definition\ of\ Survival\ Data$ \<生存数据的定义\></font>
$$\{t_i,\delta_i,x_i\}^{n}_{i=1}$$
- $t_i\quad\textit{fellowing-up time}\quad\text{跟进时间: }$
  $$
  \begin{cases}
  \text{左截断or删失：}\max\{\underset{\text{生存时间}}{T_i},\underset{\text{删失时间}}{C_i},\underset{\text{截断时间}}{Tr_i}\}\\
  \text{右截断or删失：}\min\{\underset{\text{生存时间}}{T_i},\underset{\text{删失时间}}{C_i},\underset{\text{截断时间}}{Tr_i}\}
  \end{cases}
  $$
- $\delta_i\quad state\quad\text{状态: }$
$$
\begin{cases}
\text{删失}
  \begin{cases}
    \delta_i=1&\text{if } t_i=T_i\\
    \delta_i=0&\text{if } t_i=C_i\end{cases}\\
\text{截断}
  \begin{cases}
    \delta_i=1&\text{if } t_i=T_i\\
    \delta_i=0&\text{if } t_i=Tr_i
  \end{cases}
\end{cases}$$
</br> 
- $x_i\quad covariates\quad\text{协变量: }\quad\text{weight, height, blood pressure...}$

#### <font color=Chartreuse>$\small Likelihood\ of\ Survival\ Data$ \<生存数据的似然\></font>
- $\text{似然函数：}L(\theta)=\prod_{i=1}^{n}f(x_i;\theta)=\prod_{i=1}^{n}L_i$
- $\text{生存数据的似然函数：}L(\theta)=\prod_{i=1}^{n}\Big(f(t_i;\theta)\Big)^{\delta_i}\Big(S(t_i;\theta)\Big)^{1-\delta_i}$
- $删失or截断类型不同,\ L_i\ 的形式也不同$
  $$
  \begin{cases}
    \text{正常数据}\mathit{(T_i=t_i)}&\rightarrow L_i=f(t_i;\theta)\\
    \text{右删失数据 }(T=t_{i}+)\mathit{(T_i\geq t_i)}&\rightarrow L_i=S(t_i;\theta)\\
    \text{左删失数据 }(T=t_{i}-)\mathit{(T_i\leq t_i)}&\rightarrow L_i=F(t_i;\theta)\\
    \text{区间删失数据 }\mathit{(l_i\leq T_i\leq r_i)}&\rightarrow L_i=F(r_i;\theta)-F(l_i;\theta)\\
    \text{右截断数据 }\mathit{(T_i=t_i|T<u)}&\rightarrow L_i=\cfrac{f(t_i)}{F(u)}\\
    \text{左截断数据 }\mathit{(T_i=t_i|T>u)}&\rightarrow L_i=\cfrac{f(t_i)}{S(u)}
  \end{cases}
  $$
- <font color=orange>$\text{Ps: 不完全数据举例}$</font>
  $$\begin{cases}
  右删失: 实验结束之后才发病，但无法记录具体时间\\
  左删失: 实验开始之前就发病，但不记得具体时间\\
  区间删失: 实验只在两个周一之间提取数据，\\
           \qquad\qquad\quad患者在其间发病，不知道发病确切时间\\
  右截断：实验只考虑30岁及以下的病人\\
  左截断：实验只考虑50岁及以上的病人  
  \end{cases}$$

#### <font color=Chartreuse>$\small Type\ of\ Censored\ Data$ \<删失数据的类型\></font>
$$
\begin{cases}
  \text{Type I Censoring: 删失时间固定。eg:实验只进行5天}\\
  \text{Type II Censoring: 当失败样本达到一定比例后结束实验}\\
  \text{Random Censoring: 样本删失情况是随机的}
\end{cases}
$$
- **<font color=orange>以上三种删失都不会影响似然函数结果</font>**
- **<font color=orangered>重要结论：</font>**
  $$\textit{If:}\ C\perp T|X\quad\textit{Then:}\ L_i(\theta)\propto f(t_i|\theta,X)^{\delta_i}S(t_i|\theta,X)^{1-\delta_i}$$
  <font color=Gold>$只要删失\ C\ 与事件发生时间\ T\ 无关,似然函数\ L(\theta)\ 就不受影响$</font>

---

### <font color=Aqua>$\small Kaplan\ Meier\ Estimator$ \<K-M 估计\></font>

####<font color=Chartreuse>$\small Empirical\ Distribution$ \<经验分布\></font>
- $Empirical\ Distribution:$
  $$
  F_n(x)=\cfrac{1}{n}\sum_{i=1}^n\mathbb{I}_{X_i\leq x}=
  \begin{cases}
    0,&\text{if } x\leq X_{(1)}\\
    \cfrac{k}{n}, &\text{if } X_{(k)}<x\leq X_{(k+1)}\\
    1,&\text{if } x>X_{(n)}
  \end{cases}
  $$
- 经验分布 **<font color=Gold>依概率1收敛(几乎处处收敛)</font>** 于真实分布, **<font color=orangered>(格里汶科定理)</font>**
  $$
  p\Big(\lim_{n\rightarrow\infty}F_n(x)=F(x)\Big)=1
  \implies F_n(x)\overset{a.s.}{\longrightarrow}F(x)
  $$

####<font color=Chartreuse>$\small Kaplan\ Meier\ Estimator$ \<K-M 估计\></font>
- 使用经验函数来拟合事件的分布函数 $F(t)$，进而拟合出生存函数 $S(t)$。但删失数据会给估计带来偏差。
- 引入条件概率减小偏差：
  $$
  \begin{aligned}
    S(t)&=P(T>t)\\
    \hat{S}(t)&=P(T>t_i|T>t_{i-1})\cdot \hat{S}(t_{i-1})
  \end{aligned}
  $$
- $n_j\quad j$ 时刻剩余人数 
  $d_j\quad j$ 时刻发生终点事件的人数
  $n\quad$ 总人数

  $$
  \begin{aligned}
    \because\hat{S}(t_j)&=P(T>t_j|T>t_{j-1})\cdot\hat{S}(t_{j-1})\\
    &=\cfrac{P(T>t_j,T>t_{j-1})}{P(T>t_{j-1})}\cdot\hat{S}(t_{j-1})\\
    &=\underbrace{\cfrac{n_j-d_j}{n}\div\cfrac{n_j}{n}}_\text{Empirical Distribution}\cdot\hat{S}(t_{i-1})\\
    &=\cfrac{n_j-d_j}{n_j}\cdot\hat{S}(t_{i-1})=\Big(1-\cfrac{d_j}{n_j}\Big)\cdot\hat{S}(t_{i-1})\\
    \\
    \because\hat{\lambda}_i&=\cfrac{f(t_j)}{S(t_j)}=\cfrac{f(t_j)}{P(T>t_j)}=\underbrace{\cfrac{d_j}{n}\div\cfrac{n_j}{n}}_\text{Empirical Distribution}=\cfrac{d_j}{n_j}\\
    \\
    \therefore\hat{S}(t_j)&=\prod_{j:\ t_j\leq t}(1-\hat{\lambda}_j)=\prod_{j:\ t_j\leq t}(1-\cfrac{d_j}{n_j})
  \end{aligned}
  $$
- $\text{Kaplan Meier Estimator}$ 可以有另一种视角来解释，即 **<font color=Gold>$\text{MLE}$</font>** 。
  在该视角下，似然函数最大时，$\hat{\lambda}_j\backsim B$

####<font color=Chartreuse>$\small Delta\ Method$</font>
$Assume:\ \sqrt{n}(\hat{\lambda_j}-\lambda_j)\overset{d}{\longrightarrow} N(0,\sigma^2)\\
一元\implies\sqrt{n}(g(\hat{\lambda}_j)-g(\lambda_j))\overset{d}{\longrightarrow} N(0,[g'(\lambda_j)]^2\sigma^2)\\
多元\implies\sqrt{n}(G(\hat{\Lambda}_j)-G(\Lambda_j))\overset{d}{\longrightarrow} N(0,\nabla G(\Lambda_j)^T\cdot\Sigma\cdot\nabla G(\Lambda_j))\\
\space\\
Proof:\\ 
\begin{aligned}
  &Taylor一阶展开\implies g(\hat{\lambda_j})=g(\lambda_j)+g'(\lambda_j)(\hat{\lambda_j}-\lambda_j)\\
  &\implies g(\hat{\lambda_j})-g(\lambda_j)=g'(\lambda_j)(\hat{\lambda_j}-\lambda_j)\overset{d}{\longrightarrow} N(0,[g'(\lambda_j)]^2\sigma^2)
\end{aligned}
$

####<font color=Chartreuse>$\small Confidence\ Interval\ of\ Kaplan\ Meier\ Estimator$</font>
- 利用 $Delta\ Method$ 来求出 $K-M\ Estimator$ 的置信区间
  $$
  \begin{cases}
  1.\sqrt{n}(\log \hat{S}(t)-\log S(t))\overset{d}{\longrightarrow}N(0,\sigma_t^2)\implies g(x)=\log x\\
  2.\sqrt{n}(\hat{S}(t)-S(t))\overset{d}{\longrightarrow}N(0,\sigma_t^2[S(t)]^2)\implies g(x)=x\\
  3.\sqrt{n}(\hat{S}(t)-S(t))\overset{d}{\longrightarrow}N(0,\frac{\sigma_t^2}{[\log S(t)]^2})\implies g(x)=\log[-\log x]
  \end{cases}
  $$
- $\sigma_t^2=n\sum_{t_j<t}\cfrac{d_j}{n_j(n_j-d_j)}$
- 置信区间：
  $$-N_{\alpha/2}\leq\cfrac{g[\hat{S}(t)]-g(Survival\ Rate)}{\sqrt{Var\Big[g(\hat{S}(t))\Big]}}\leq N_{\alpha/2}$$

---

### <font color=Aqua>$\small Group\ Testing$ \<分组检验\></font>
####<font color=Chartreuse>$\small Contingency\ Table$ \<列联表\></font>
- 展示的为行数 $R$、列数 $C$ 都为2时的列联表, <font color=Gold>行数列数可增加</font>
$$
\def\arraystretch{1.5}
  \begin{array}{c|c|c|c}
       & C_1 & C_2 & Total & \\ \hline
    R_1 & a & b & a+b & \\ \hline
    R_2 & c & d & c+d & \\ \hline
    Total& a+c & b+d & a+b+c+d=n & 
  \end{array}
$$

####<font color=Chartreuse>$\small Pearson\enspace \chi^2\enspace Test$ \<皮尔逊卡方检验\></font>
- 皮尔逊卡方检验常检验 **<font color=Gold>列联表的拟合优度$_1$</font>** 和 **<font color=Gold>列联表行列是否相关$_2$</font>**
- 利用了 **<u>大样本下渐近卡方分布的性质</u>**
- $1\implies$ $H_0:理论频数与实际频数相同$
  $2\implies$ $H_0:行列因素相互独立$
  统计量：
  $$
  \chi^2=\sum_{i=1}^{R}\sum_{j=1}^{C}\cfrac{(f_o-f_e)^2}{f_e}
  $$
  $$
  \textit{if:}\enspace\chi^2>\chi_{\alpha}^2(df)\implies\textit{Then:}\enspace refuse\ H_0
  $$
  $
  f_o：\text{实际频数}\enspace Eg: f_o=a\\
  F_e: \text{理论频数}\enspace Eg: f_e=\cfrac{(a+b)}{n}\cdot\cfrac{(a+c)}{n}\cdot n=\cfrac{(a+b)(c+d)}{n}\\
  df: \text{自由度}\enspace df=(R-1)\times(C-1)
  $
</br>
- 两个问题其实是等价的。若行列因素相互独立$_2$，则可通过相互独立两事件的积事件的定义 <font color=Gold>$P(AB)=P(A)P(B)$</font> 由边际分布求出联合概率密度。同时验证了同样用此方法求出的理论频数的正确性$_1$。

####<font color=Chartreuse>$\small Fisher's\  Exact\enspace Test$ \<Fisher确切概率法\></font>
- 利用了 **<u>超几何分布的定义</u>**
- $Fisher$ 确切概率法直接计算出 $p-value$
  $$
  \begin{aligned}
    p&=\frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}=\frac{\binom{a+b}{b}\binom{c+d}{d}}{\binom{n}{b+d}}\\&=\frac{(a+b)！(c+d)！(a+c)！(b+d)！}{a！b！c！d！n！}
  \end{aligned}
  $$
  $$
  \textit{if:}\enspace p<\alpha
  \implies\textit{Then:}\enspace refuse\ H_0
  $$
- 若为多行多列数据，可使用 **<u>多元超几何分布</u>**
- Fisher确切概率法使用条件：
  <font color=Gold>$
  \textbf{某个格子的理论频数}\; 
  \bm{f_e<5}\; 
  \textbf{或}\;
  \bm{p-value\approx\alpha}\;\textbf{时}\\
  \textbf{使用Fisher确切概率法}
  $</font>

####<font color=Chartreuse>$\small McNemar's \enspace Test$ \<McNemar检验\></font>
- McNemar检验用来检验 **<font color=Gold>成对数据</font>** 。即同一样本采用不同方法产生的数据,且只能在使用 $2\times2$ 列联表上使用 
  $$
\def\arraystretch{1.5}
  \begin{array}{c|c|c|c}
    & Test1\ pos & Test1\ neg & Total & \\ \hline
    Test2\ pos  & a & b & a+b & \\ \hline
    Test2\ neg  & c & d & c+d & \\ \hline
    Total& a+c & b+d & a+b+c+d & 
  \end{array}
$$

  
- $H_0:两方法效果相同\implies Assume：\begin{cases}a+c=a+b\\b+d=c+d\end{cases}\\\implies b=c$
- 统计量：
  $$\chi^2=\cfrac{(b-c)^2}{b+c}$$
  $$
  \textit{if:}\enspace\chi^2>\chi_{\alpha}^2(1)\implies\textit{Then:}\enspace refuse\ H_0
  $$
- **<font color=Gold>此时，两组数据之间不相互独立的，因此皮尔逊 $\chi^2$ 统计量失效。使用 $\cfrac{b+c}{2}$ 来作为理论频数，并使用 $\chi^2$ 统计量的计算方法即可得出此统计量。并证明 $b=c$</font>**
- <font color=Orangered>$\textbf{Binomial}\overset{limit}{\longrightarrow}\textbf{Poisson}\overset{limit}{\longrightarrow}\textbf{Normal}$</font>
####<font color=Chartreuse>$\small Cochran-Mantel-Haenszel\enspace Test$ \<分层卡方检验\></font>
$$
\def\arraystretch{1.5}
  \begin{array}{c|c|c|c}
    Layer\ \textcolor{aqua}{\bm{i}}& Treat & Non-treat & Total & \\ \hline
    Case  & a_i & b_i & a_i+b_i & \\ \hline
    Control  & c_i & d_i & c_i+d_i & \\ \hline
    Total& a_i+c_i & b_i+d_i & a_i+b_i+c_i+d_i=n_i & 
  \end{array}
$$
- CMH检验常用在多中心实验中，以去除实验中的中心效应。
- 几率 $Odds$ : 是一种替代概率的概念，它没有上下界，因此可以用多种模型对其进行拟合。
  $$
  Odds(A)=\cfrac{P(A)}{P(\bar{A})}=\cfrac{P(A)}{1-P(A)}
  $$
- 优势比 $OR$ : 在不受实验组和对照组比例的条件下，得到因果之间的联系。
  $$
  OR=\frac{\cfrac{\frac{a_i}{a_ic_i}}{\frac{c_i}{a_ic_i}}}{\cfrac{\frac{b_i}{b_id_i}}{\frac{d_i}{b_id_i}}}=\frac{\cfrac{a_i}{c_i}}{\cfrac{b_i}{d_i}}=\cfrac{a_id_i}{b_ic_i}
  $$
- CMH检验使用多层优势比来去除中心效应。
  $$
  OR=\cfrac{\sum_{i=1}^T\cfrac{a_id_i}{n_i}}{\sum_{i=1}^T\cfrac{b_ic_i}{n_i}}
  $$
- $H_0:OR=1\;结局与治疗方案无关$
- 统计量  
  $$
  \chi^2=\cfrac{\Big[\sum_{i=1}^T(a_i-\cfrac{(a_i+b_i)(a_i+c_i)}{n_i})\Big]^2}{\cfrac{(a_i+b_i)(a_i+c_i)(b_i+c_i)(b_i+d_i)}{n_i^2(n_i-1)}}
  $$
  $$\textcolor{gold}{\textbf{相当于对}a_i\textbf{进行了Z-Score 标准化}}$$
  $$
  \textit{if:}\enspace\chi^2>\chi_{\alpha}^2(1)\implies\textit{Then:}\enspace refuse\ H_0\ \&\ calculate\ OR
  $$
- 显然，此时 $a_i$ 服从<font color=Gold>超几何分布</font>; 且注意，<font color=Gold>此时 $df=1$ 是因为累加的是各实验中心的样本数，而并非单个特征的类数 (行数 or 列数)，只有行列增加时自由度是增加的，只增加样本数不改变自由度</font>。
- 流程：进行CMH检验 $\longrightarrow$ 计算多层优势比 $OR$

####<font color=Chartreuse>$\small Log-Rank\enspace Test$ \<Log-Rank 检验\></font>
- $
  \begin{cases}
  H_0:S_1=S_2&Group=2\\
  H_0:S_1=\cdots=S_n&Group=n
  \end{cases}
  \\ 
  \textcolor{gold}{\textbf{与下式等价}}\\
  \begin{cases}
  H_0:\lambda_1(t)=\lambda_2(t)&Group=2\\
  H_0:\lambda_1(t)=\cdots=\lambda_n(t)&Group=n
  \end{cases}
  $
- $Group=2$
  - 需指出, $Group=2$ 时, $\text{Log-Rank test与CMH test 相同}$ 
  $$
  \def\arraystretch{1.5}
  \begin{array}{c|c|c|c}
    Layer\ \textcolor{aqua}{\bm{t}}& Control & Case & Total & \\ \hline
    Treat  & d_{1t} & d_{2t} & d_t & \\ \hline
    Non-treat  & n_{1t}-d_{1t} & n_{2t}-d_{2t} & n_t-d_t & \\ \hline
    Total& n_{1t} & n_{2t} & n_t & 
  \end{array}
  $$
  - 此时，将存在终点事件的不同时间点 $t$ 看作 $CMH$ 检验中不同的层。
  - 统计量
    $$
    \begin{aligned}
    \chi^2&=\frac{W^2}{V}\\
    W&=\sum_{t=1}^Tw_t; V=\sum_{t=1}^Tv_t\\
    w_t&=d_{1t}-n_{1t}\frac{d_t}{n_t}\\
    v_t&=\frac{n_{1t}n_{2t}d_t(n_t-d_t)}{n_t^2(n_t-1)}
    \end{aligned}\\
    \textit{if:}\enspace\chi^2>\chi_{\alpha}^2(1)\implies\textit{Then:}\enspace refuse\ H_0
    $$
  - 该统计量先对 $d_{1t}$ 进行 Z-Score 标准化；则 $w_t$ 渐进服从正态分布，再使用正态分布的可加性即可证明 $W$服从正态分布，将其除于 $V$ 后，即可证得 $\chi^2$ 渐进服从 $\chi^2$ 分布。
-  $Group=n$
   -  $\text{Log-Rank test} 在\ Group=n\ 时的情况将 \text{CMH test} \\拓展到了多分类的情况$
   - 统计量
    $$
    \begin{aligned}
    \chi^2&=W^TV^{-1}W\\
    W&=\sum_{t=1}^Tw_t; V=\sum_{t=1}^TV_t\\
    w_t&=\Big(d_{1t}-n_{1t}\frac{d_t}{n_t},\cdots,d_{pt}-n_{pt}\frac{d_t}{n_t}\Big)\\
    \end{aligned}
    $$$$
    \begin{cases}
    (V_t)_{ii}&=\cfrac{(n_j-n_{it})n_{it}d_t(n_t-d_t)}{n_t^2(n_t-1)}&\text{对角线元素}\\
    (V_t)_{ij}&=\cfrac{n_{it}n_{jt}d_t(n_t-d_t)}{n_t^2(n_t-1)}&\text{非对角线元素}
    \end{cases}
    \\
    \quad\\
    \textit{if:}\enspace\chi^2>\chi_{\alpha}^2(p-1)\implies\textit{Then:}\enspace refuse\ H_0
    $$
- $\textcolor{gold}{\textbf{Weighted Log-Rank Test}}$
  - 为了使 **<font color=orangered>样本多的时候更大的权重</font>** 而引入。
  $$
  W=\sum_{t=1}^T\alpha_tw_t;\quad V=\sum_{t=1}^T\alpha_t^2v_t\\
  \quad\\
  \begin{aligned}
  &\alpha_t=n_j&\implies &\text{Gehan-Breslow Test}\\
  &\alpha_t=\hat{S}(t)&\implies &\text{Peto-Prentice Test}\\
  &\alpha_t=\Big(\hat{S}(t)\Big)^{\rho}&\implies &\text{Peto-Prentice Test}\\
  \end{aligned}
  $$

- $\textcolor{gold}{\textbf{Stratified Log-Rank Test}}$
  - 为了 **<font color=orangered>控制不同变量</font>** 而引入。
  - $H_0: S(t|k)=S(t|k)\\ \text{在固定因素k的情况下二者是否相等}$
  - 统计量：
    $$
    \chi^2=\frac{W^2}{V}=\cfrac{(\sum_k\sum_tw_{tk})^2}{\sum_tv_{tk}}
    \\
    \quad\\
    \textit{if:}\enspace\chi^2>\chi_{\alpha}^2(p-1)\implies\textit{Then:}\enspace refuse\ H_0
    $$

---

### <font color=Aqua>$\small Likelihood\ Testing$ \<似然检验\></font>
####<font color=Chartreuse>$\small Score\ Function\ \&\ Fisher\ Information$</font>
- 得分向量与 $\text{Fisher}$ 信息阵与似然函数相关，下面将给出相关定义。
  - 单参数 
  $$
  \begin{aligned}
    \text{Likelihood Function: }\quad L(\theta)&=\prod_ip(x_i,\theta)\\
    l(\theta)&=\ln L(\theta)\\
    \text{Score Function: }\quad U(\theta)&=\cfrac{\partial}{\partial\theta}l(\theta)\\
    \text{Fisher Information: }\quad I(\theta)&=-E_X(\cfrac{\partial}{\partial\theta}U(\theta))\\&=-E_X(\cfrac{\partial^2}{\partial\theta^2}l(\theta))
  \end{aligned}
  $$
  - 多参数 
  $$
  \begin{aligned}
    \text{Likelihood Function: }\quad L(\vec{\theta})&=\prod_ip(x_i,\vec{\theta})\\
    l(\vec{\theta})&=\ln L(\vec{\theta})\\
    \text{Score Function: }\quad U(\vec{\theta})&=\cfrac{\partial}{\partial\theta_i}l(\vec{\theta})\\
    &=\nabla_{\theta}l(\vec{\theta})\\
    \text{Fisher Information: }\quad I(\vec{\theta})&=-E_X(\cfrac{\partial}{\partial\theta_j}U(\vec{\theta}))\\
    &=-E_X(\nabla_{\theta}U(\vec{\theta}))
    \\&=-E_X(\cfrac{\partial^2}{\partial\theta_i\theta_j}l(\vec{\theta}))
    \\&\text{(Matrix)}
  \end{aligned}
  $$

- 下面给出几条 **<font color=Gold>重要结论</font>** ：
  1. $U(\theta)$ 其实也是关于 $X$ 的函数
  2. $E_X\Big[U(\theta)\Big]=0$
  3. $Var_X\Big[U(\theta)\Big]=I(\theta)$
  4. $\text{if: X is iid}\implies \\U(\theta)=\sum_iU(x_i;\theta)\;\&\;I(\theta)=\sum_iI_i(\theta)$
  5. 在计算时，可以使用近似值 $\tilde{I}(\theta)$ 替代 $I(\theta)$
  $\tilde{I}(X)\overset{E(x)=\frac{1}{n}\sum_xx_i}{=\!=\!=\!=\!=\!=\!=}\cfrac{\partial}{\partial\theta}U(-E_X(X);\theta)$

####<font color=Chartreuse>$\small Score\ Test$</font>
- $H_{0}: \theta=\theta_{0}$
- 统计量：<font color=Gold>(理论来自于 $U(x;\theta)$ 的渐进分布)</font>
  $$
  \begin{aligned}
    \cfrac{U(\theta_0)}{\sqrt{I(\theta_0)}}&\overset{d}{\longrightarrow}N(0,1)\quad\text{单参数}\\
    U(\vec{\theta_0})^TI^{-1}(\vec{\theta_0})\ U(\vec{\theta_0})&\overset{d}{\longrightarrow}\chi^2(p)\quad\text{多参数}
  \end{aligned}
  $$

####<font color=Chartreuse>$\small Wald\ Test$</font>
- $H_{0}: \theta=\theta_{0}$
- 统计量：<font color=Gold>(理论来自于 $U(x;\theta)$ 的 Taylor 一阶展开)</font>
  $$
  \begin{aligned}
    \cfrac{(\hat{\theta}-\theta_0)}{\sqrt{I(\theta)}}&\overset{d}{\longrightarrow}N(0,1)\quad\text{单参数}\\
    (\hat{\theta}-\vec{\theta_0})^TI^{-1}(\hat{\theta})\ (\hat{\theta}-\vec{\theta_0})&\overset{d}{\longrightarrow}\chi^2(p)\quad\text{多参数}
  \end{aligned}
  $$
- $U(\theta)$ 的 Taylor 一阶展开:
  $$U(\theta)\approx U(\hat{\theta})-I(\hat{\theta})(\theta-\hat{\theta})\\
  \implies I(\hat{\theta})(\theta-\hat{\theta})\overset{d}{\longrightarrow} N(0,I(\hat{\theta}))$$
####<font color=Chartreuse>$\small Likelihood\ Ratio\ Test$</font>
- $H_{0}: \theta=\theta_{0}$
- 统计量：<font color=Gold>(理论来自于 $l(\theta)$ 的 Taylor 二阶展开)</font>
   $$
    -2(l(\theta_0)-l(\hat{\theta}))\overset{d}{\longrightarrow}\chi^2(p)
  $$
- $l(\theta)$ 的 Taylor 二阶展开:
  $$
  l(\theta)=l(\hat{\theta})+U(\hat{\theta})(\theta-\hat{\theta})-\frac{1}{2}I(\hat{\theta})(\theta-\hat{\theta})^2\\
  \implies U(\hat{\theta})(\theta-\hat{\theta})=0\implies\\
  l(\theta)=l(\hat{\theta})-\frac{1}{2}I(\hat{\theta})(\theta-\hat{\theta})^2\\
  \implies I(\hat{\theta})(\theta-\hat{\theta})^2\overset{d}{\longrightarrow} N(0,I(\hat{\theta}))\\
  \implies (\theta-\hat{\theta})^TI(\hat{\theta})^{-1}\ (\theta-\hat{\theta})\overset{d}{\longrightarrow}\chi^2(p)
  $$
#### <font color=Gold>三种检验方法在大样本情况下渐近等价</font> 

---

### <font color=Aqua>$\small Proportional\ Hazard Model$ \<比例风险模型\></font>

####<font color=Chartreuse>$\small Newton-Raphson\  method$\<牛顿迭代法\></font>
- 在比例风险模型中，由于指数函数的存在，使我们难以使用最小二乘来进行回归。因此我们将使用极大似然估计的方法对比例风险模型进行求解。
  $$
  \text{MLE}\implies\argmax_{\theta} L(\theta)\implies\cfrac{\partial}{\partial\theta}\ln L(\theta)=0\implies U(\theta)=0
  $$
  $$
  \text{进行Taylor展开: }\quad0=U(\theta)\approx U(\hat{\theta})-I(\hat{\theta})(\theta-\hat{\theta})\\
  \implies \hat{\theta_{i+1}}\coloneqq\hat{\theta_i}+I^{-1}(\hat{\theta_i})U(\hat{\theta_i})
  $$

####<font color=Chartreuse>$\small Exponential\ Distribution$\<指数分布\></font>
- <font size=6>$\text{pdf: } p(t)=\lambda e^{-\lambda t},\quad\text{for }t\geq 0$</font>

####<font color=Chartreuse>$\small Weibull\ Distribution$\<韦布尔分布\></font>
- <font size=6>$\text{pdf: } p(t)=\gamma\lambda(\lambda t)^{\gamma-1}e^{-(\lambda t)^{\gamma}},\quad\text{for }t\geq 0$</font>

####<font color=Chartreuse>$\small Extreme\  Value\ Distribution$\<极值分布\></font>

####<font color=Chartreuse>$\small Proportional\ Hazard Model$\<指数回归模型\></font>
- 形式：
  $$
  \lambda_i(t)=\lambda(t)\exp\{x_i^T\beta\},\quad\lambda(t)\geq 0
  $$

####<font color=Chartreuse>$\small Exponential\ Regression\ Model$\<指数回归模型\></font>
- 形式：
  $$
  \lambda_i(t)=\lambda\exp\{x_i^T\beta\}
  $$
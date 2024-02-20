<font size=5>

## <font color=Gold>$Generalize\ Linear\ Model$</font>

### <font color=Chartreuse>$N-R\ Method$</font>
- $The\ way\ to\ solve\ generalized\ linear\ model\;(GLM)$
- $Use\ most\ likelihood\ estimator\;(MLE)\ to\  solve\;(GLM)$
$$\theta^{(i+1)}\coloneqq\theta^{(i)}+I^{-1}(\theta^{(i)})U(\theta^{(i)})$$
- $U(\theta)\ is\ Score\ Function$
$$
\begin{aligned}
U(\theta)&=\cfrac{\partial}{\partial\theta_i}\ln L(\theta)\\
&=\nabla_{\theta}\ln L(\theta)
\end{aligned}
$$
- $I(\theta)\ is\ Fisher\ Information\ Matrix\;(FIM)$
$$
\begin{aligned}
I(\theta)&=-E\Big[\cfrac{\partial}{\partial\theta_i}U(\theta)\Big]\\
&=-E\Big[\nabla_{\theta}U(\theta)\Big]
\end{aligned}
$$
### <font color=Chartreuse>$Poisson\ Regression$</font>
#### <font color=Aqua>$Poisson\ $</font>  

### <font color=Chartreuse>$Negative\ Binomial\ Regression$</font>

### <font color=Chartreuse>$Poisson\ Regression$</font>
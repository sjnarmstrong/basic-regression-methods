---
bibliography:
- 'reporttemplate.bib'
---

We have seen how Bayesian methods can be useful in determining the
probability of certain events occurring. We now turn to the problem of
linear regression. This involves determining the process which is used
to generate a set of target values given a set of input variables.
Formally, given a set of input variables
$\mathbf{X}\equiv\left(\mathbf{x}_1,...,\mathbf{x}_N\right)^T$ and a set
of target variables
$\mathbf{T}\equiv\left(\mathbf{t}_1,...,\mathbf{t}_N\right)^T$, we seek
to find the function $f(\mathbf{x})$, which was used to generate the
target variables from the given input. This is a complex task as there
is often an element of noise added to the function $f(\mathbf{x})$
before the target variable is produced. To keep things simple, we will
focus on data containing a 1D input variable $x$ and a 1D output target
$t$. furthermore, we turn to a polynomial function, in order to
approximate our underling function $f(\mathbf{x})$. This is due to the
fact that most functions can be accurately approximated by a few terms
of their Taylor expansion. The parametric function used can therefore be
written as follows: $$\begin{aligned}
y(x,\mathbf{w}) &= \sum_{j=0}^{M} w_jx^j\\
&= \mathbf{w}^T\mathbf{\phi}(x)\\
\end{aligned}$$ Where $M$ is the order of the polynomial function and
plays a role in the maximum complexity of the function. We have also
defined $\mathbf{\phi}(x)$ as $\left(x^0,x^1,...,x^M\right)^T$. Further,
for convenience when handling multiple data-points, we define $\Phi$ as
$\left(\mathbf{\phi}(x_1),...,\mathbf{\phi}(x_N)\right)^T$. Here $N$
denotes the number of data-points.\
We can now see that in order to approximate the given function
$f(\mathbf{x})$, we must determine suitable values for the parameters
$\mathbf{w}$. In this exercise we explore the use of Bayesian and
classical methods to achieve this.

### Least Squares Approach

The least squares approach tries to minimise the squared error between
the target variables and the parametrised function. This error function
is defined as follows: $$\begin{aligned}
E(\mathbf{w})&=\frac{1}{2}\sum_{i=1}^{N}\left(t_i-y(x_i,\mathbf{w})\right)^2\\
&=\frac{1}{2}\sum_{i=1}^{N}\left(t_i-\mathbf{w}^T\mathbf{\phi}(x_i)\right)^2
\end{aligned}$$ This is then minimised in close form by taking the
derivative with respect to W and setting this to zero.
$$\frac{\partial E(\mathbf{w})}{\partial \mathbf{w}}=-\sum_{i=1}^{N}\left(t_i-\mathbf{w}^T\mathbf{\phi}(x_i)\right)\mathbf{\phi}(x_i)=0$$

The result is given by [@christopher2016pattern] in equation 3.15 as:
$$\Phi^T\Phi\mathbf{w}=\Phi^T\mathbf{t}
\label{eqn:E3:Wml}$$

### Maximum Likelihood Approach

We now consider the likelihood of obtaining the target data from the
parametric function. For this we assume that the data has a Gaussian
distribution around the given function at any given input $x$. This is
therefore written as follows:
$$p(\mathbf{t}|\mathbf{X},\mathbf{w},\beta)=\prod_{i=1}^{N}\mathcal{N}\left(t_n|\mathbf{w}^T\mathbf{\phi}(x_i),\beta^{-1}\right)$$
Where $\beta$ is the precision of the Gaussian distribution. Taking the
natural logarithm of this function we get:
$$\text{ln}\left(p(\mathbf{t}|\mathbf{X},\mathbf{w},\beta)\right)=\frac{N}{2}\text{ln}(\beta)-\frac{N}{2}\text{ln}(2\pi)-\frac{\beta}{2}\sum_{i=1}^{N}\left(t_i-y(x_i,\mathbf{w})\right)^2
\label{eqn:E3:MLErr}$$ Minimising the log likelihood is identical to
minimising the sum of squares error. This can be seen in equation
\[eqn:E3:MLErr\], as the only term dependant on $\mathbf{w}$ is a scalar
multiple of the least squares error function. Due to this,
$\mathbf{w}_{ML}$ can be determined with the use of equation
\[eqn:E3:Wml\].

### Bayesian Approach

The Bayesian approach attempts to determine the probability of the
parameters $\mathbf{w}$ given the target variables $\mathbf{t}$.
Assuming this takes a Gaussian form, we can model this probability as
follows:
$$p(\mathbf{w}|\mathbf{t})=\mathcal{N}\left(\mathbf{w}|\mathbf{m}_N,\mathbf{S}_N\right)$$
Where $\mathbf{m}_N$ represents the mean of the weights and
$\mathbf{S}_N$ represents the variance. These can be determined in a
Bayesian approach by assuming an initial $\mathbf{m}_0$ and
$\mathbf{S}_0$. Equations 3.50 and 3.51 from [@christopher2016pattern]
can then be used to update these parameters. This update step is given
as follows: $$\begin{aligned}
\mathbf{S}_N^{-1}&=\mathbf{S}_0^{-1}+\beta\Phi^T\Phi\\
\mathbf{m}_N&=\mathbf{S}_N\left(\mathbf{S}_0^{-1}\mathbf{m}_0+\beta\Phi^T\mathbf{t}\right)\label{eqn:E3:mn}\end{aligned}$$

It is common practice to assume a zero mean for $\mathbf{m}_0$ and a
large variation for $\mathbf{S}_0$ corresponding to $\alpha \mathbf{I}$.
Here $\mathbf{I}$ is known as the identity matrix.

### Results and Discussion of above methods {#sec:E3:ResDesc}

We now run the above algorithms on a dataset containing 10 points with
corresponding $x$ and $t$ values. We will assume that the data is
generated in such a manner that $\beta=11.1$ and
$\frac{1}{\alpha}=5\times 10^{-3}$. Furthermore, we will assume a zero
prior mean on the weights for the Bayesian linear regression. We start
by considering an order 4 polynomial function. The results are given in
figure \[fig:E3:Or4:LSQ\].\

![Plot of the results of least squares curve fitting (left) and maximum
likelihood (right), with a order 4 polynomial
function.[]{data-label="fig:E3:Or4:LSQ"}](Figs/Q3/Q3P1Order_4 "fig:"){width="49.00000%"}
![Plot of the results of least squares curve fitting (left) and maximum
likelihood (right), with a order 4 polynomial
function.[]{data-label="fig:E3:Or4:LSQ"}](Figs/Q3/Q3P2Order_4 "fig:"){width="49.00000%"}

One can see that these produce identical results as they are
mathematically equivalent. With the Bayesian approach, we are able to
quantify our certainty of a predicted point. This is shown in figure
\[fig:E3:Or4:Bays\], by plotting the standard deviation around the mean
indicated by the dashed line. It is also useful as it is a generative
function. This means that we are able to produce various data-points
following a similar distribution to the observed data-points. It is also
possible to generate a list of plots that are lightly to have generated
the data. This is done in the right plot of figure \[fig:E3:Or4:Bays\].\

![Plot of the results of Bayesian curve fitting, with a order 4
polynomial
function.[]{data-label="fig:E3:Or4:Bays"}](Figs/Q3/Q3P3_M4 "fig:"){width="49.00000%"}
![Plot of the results of Bayesian curve fitting, with a order 4
polynomial
function.[]{data-label="fig:E3:Or4:Bays"}](Figs/Q3/Q3P3RandomSample_M4 "fig:"){width="49.00000%"}

We now fit the graphs for $M=9$. The results of the least squares and
maximum likelihood curve fitting is shown in figure \[fig:E3:Or9:LSQ\].
These graphs show a phenomenon known as over-fitting. The data has 10
degrees of freedom, all of which can be accounted for by the parametric
equation. Due to this, the best fit for the data, is one that goes
through all the points. This has a very low error function but often
does not generalise well to new data. Assuming that the test data was
generated from a sine function, one can see that these new functions
provide a poor approximation.\

![Plot of the results of least squares curve fitting (left) and maximum
likelihood (right), with a order 9 polynomial
function.[]{data-label="fig:E3:Or9:LSQ"}](Figs/Q3/Q3P1Order_9 "fig:"){width="49.00000%"}
![Plot of the results of least squares curve fitting (left) and maximum
likelihood (right), with a order 9 polynomial
function.[]{data-label="fig:E3:Or9:LSQ"}](Figs/Q3/Q3P2Order_9 "fig:"){width="49.00000%"}

The results of Bayesian regression are less effected by the change in
order and one can hardly identify the difference between order 4 and
order 9. This is due to an inherent feature of Bayesian regression,
whereby one can identify over-fitting with the training data alone. This
mechanism can be intuitively understood by referring to equation 3.55
from [@christopher2016pattern]. This states:
$$\text{ln}\left(p(\mathbf{w}|\mathbf{t})\right)=\frac{\beta}{2}\sum_{i=1}^{N}\left(t_n-y(x_i,\mathbf{w})\right)^2-\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}+\text{const}.$$

From the term $\frac{\alpha}{2}\mathbf{w}^T\mathbf{w}$, it is possible
to see that $\text{ln}\left(p(\mathbf{w}|\mathbf{t})\right)$ is
negativity influenced by adding more parameters. Due to this, the
Bayesian regression function will limit its effective complexity to keep
the effects of this term low.

![Plot of the results of Bayesian curve fitting, with a order 9
polynomial
function.[]{data-label="fig:E3:Or9:Bays"}](Figs/Q3/Q3P3_M9 "fig:"){width="49.00000%"}
![Plot of the results of Bayesian curve fitting, with a order 9
polynomial
function.[]{data-label="fig:E3:Or9:Bays"}](Figs/Q3/Q3P3RandomSample_M9 "fig:"){width="49.00000%"}

It is also interesting to see how the standard deviation of the graphs
change as the number of available training points are reduced. This is
shown in figure \[fig:E3:Or9:Bays:RandRem\]. Here, 5 points have been
removed from near the start of the data. Due to the lack of information,
The standard deviation of the function around that region is increased.
This result is very useful for real life applications, where the
certainty of the predictions is required to make an informed decision.

![Plot of the results of Bayesian curve fitting on partial data, with a
order 9 polynomial
function.[]{data-label="fig:E3:Or9:Bays:RandRem"}](Figs/Q3/Q3P3_minus_a_few_points_M9 "fig:"){width="49.00000%"}
![Plot of the results of Bayesian curve fitting on partial data, with a
order 9 polynomial
function.[]{data-label="fig:E3:Or9:Bays:RandRem"}](Figs/Q3/Q3P3RandomSample_minus_a_few_points_M9 "fig:"){width="49.00000%"}

### Bayesian Model Comparison {#sec:E3:ModComp}

We now use Bayesian methods to determine the best model $\mathcal{M}$
out of a set of models to explain the underling data $\mathcal{D}$. For
this we need to evaluate $p(\mathcal{M}_i|\mathcal{D})$. For this, we
can use bays rule which states:
$$p(\mathcal{M}_i|\mathcal{D})=\frac{p(\mathcal{D}|\mathcal{M}_i)p(\mathcal{M}_i)}{p(\mathcal{D})}
\label{eqn:E3:MgD}$$ If we assume that the prior probability
$p(\mathcal{M}_i)$ is constant over all models. Then we can simplify
equation \[eqn:E3:MgD\] to
$$p(\mathcal{M}_i|\mathcal{D})=p(\mathcal{D}|\mathcal{M}_i)\times \text{Const}
\label{eqn:E3:EVeqProb}$$ Therefore, it is equivalent to work out the
$p(\mathcal{D}|\mathcal{M}_i)$ and normalise over all the models.
Therefore, when comparing a list of polynomial functions, we can use
$p(\mathbf{t}|\mathbf{w},\alpha,\beta)$ to determine the best model for
the data. This is known as the evidence function. The formula required
to calculate this is given by [@christopher2016pattern] in equation
3.78. This states that:
$$p(\mathbf{t}|\mathbf{w},\alpha,\beta)=\left(\frac{\beta}{2\pi}\right)^{N/2}\left(\frac{\alpha}{2\pi}\right)^{M/2}\int\text{exp}\left\lbrace-E(\mathbf{w})\right\rbrace d\mathbf{w}$$
Where we can use equation 3.85 from [@christopher2016pattern], which
states:
$$\int\text{exp}\left\lbrace-E(\mathbf{w})\right\rbrace d\mathbf{w}=\text{exp}\left\lbrace-E(\mathbf{m}_N)\right\rbrace(2\pi)^{M/2}\left|\mathbf{A}\right|^{-1/2}$$
In order to compute $p(\mathbf{t}|\mathbf{w},\alpha,\beta)$, we also
require the following: $$\begin{aligned}
    &E(\mathbf{m}_N)=\frac{\beta}{2}\left\|\mathbf{t}-\Phi\mathbf{m}_N\right\|^2+\frac{\alpha}{2}\mathbf{m}_N^T\mathbf{m}_N\\
    &A=\alpha\mathbf{I}+\beta\Phi^T\Phi\end{aligned}$$ Using a new
dataset containing 80 samples produced in a similar fashion to the
dataset used in section \[sec:E3:ResDesc\], we can evaluate the evidence
as given above. This is done for $M=0$ up until $M=9$. The results of
this are shown in figure \[fig:E3:Evi\].

![Plot of the model evidence for various values of
M.[]{data-label="fig:E3:Evi"}](Figs/Q3/Q3P6Evidance){width="60.00000%"}

We can see from this that the best for to the data relates to $M=3$. To
justify this result, we can turn to the Taylor expansion of a sine
function. This is given by:
$$sin(x)=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+...$$ This is an
odd function and hence even powers of $x$ do not contribute to the final
form of the function. Furthermore, the factorial in the denominator of
each term, means that the contribution of each term diminishes quickly.
These observations can be seen in figure \[fig:E3:Evi\] as $M=3$ is a
clear maximum followed by a sudden and sharp drop in preceding terms.
The plot corresponding to the most likely model is given in figure
\[fig:E3:m3:DTA2\].\

![Plot of the fitted curve for
M=3[]{data-label="fig:E3:m3:DTA2"}](Figs/Q3/Q3P6_Order3){width="60.00000%"}

### Bayesian Model Averaging

We now average all the models tested in section \[sec:E3:ModComp\]. For
this, we can take a weighted sum over the model space. This is given by
equation 3.67 from [@christopher2016pattern].
$$p(y|\mathbf{x},\mathcal{D})=\gamma\sum_{i=0}^{M_{max}}p(t|x,\mathcal{M_i},\mathcal{D})p(\mathcal{M_i}|\mathcal{D})$$
We can determine $p(\mathcal{M_i}|\mathcal{D})$ by normalising over the
evidence function due to equation \[eqn:E3:EVeqProb\]. We can then use
equations \[eq:E3:muave\] and \[eq:E3:sigmaave\], provided by
[@trailovic2002variance], to determine the mean and standard deviation
of the weights. Note that these equations only estimate the mean and
variance of the distribution. This is due to the fact that a mixture
distribution will most likely be multi-modal and contain more that one
local maximum.

$$\begin{aligned}
&\mu(x)=\sum_{i=0}^{M_{max}}p(\mathcal{M_i}|\mathcal{D})\mu_i(x)\label{eq:E3:muave}\\
&\sigma^2(x)=\sum_{i=0}^{M_{max}}p(\mathcal{M_i}|\mathcal{D})((\mu_i(x)-\mu(x))^2+\sigma_i^2(x))\label{eq:E3:sigmaave}\end{aligned}$$

The results of the mixture distribution are given in figure
\[fig:E3:mix\].

![Plot of the fitted curve for a weighted mixture
distribution[]{data-label="fig:E3:mix"}](Figs/Q3/Q3P6Mix.png){width="60.00000%"}

When comparing figures \[fig:E3:m3:DTA2\] and \[fig:E3:mix\], one can
see that the two are very similar. Hence it would be valid to use the
most likely model as an approximation to the mixture distribution. This
saves a lot of computation power and not much accuracy is lost in doing
so.

### Determining the hyperparameters

In order to determine $\alpha$ and $\beta$, we first need to assume an
initial $\alpha$ and $\beta$. We then compute $\mathbf{m}_N$ using
equation \[eqn:E3:mn\] with the initial guess of $\alpha$ and $\beta$.
We then need to compute the following two values: $$\begin{aligned}
    &E_w(\mathbf{m}_N)=\frac{1}{2}\mathbf{m}_N^T\mathbf{m}_N\\
    &E_d(\mathbf{m}_N)=\frac{1}{2}\sum_{i=1}^{N}\left\lbrace t_i-\mathbf{m}_N^T\mathbf{\phi}(x_i)\right\rbrace^2\end{aligned}$$
This is then used to compute the new parameters $\alpha$ and $\beta$
using equations 3.98 and 3.99 from [@christopher2016pattern].
$$\begin{aligned}
&\alpha=\frac{M}{2E_w(\mathbf{m}_N)}\\
&\beta=\frac{N}{2E_d(\mathbf{m}_N)}\end{aligned}$$ This is then repeated
until convergence or until a maximum number or iterations is reached. It
is important to note that this method is only valid when the number of
data points is greatly larger than the order of the polynomial function.
If this is not the case, one must employ a more complicated procedure.

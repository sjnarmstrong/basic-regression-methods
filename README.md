---
bibliography:
- 'reporttemplate.bib'
---

We have seen how Bayesian methods can be useful in determining the
probability of certain events occurring. We now turn to the problem of
linear regression. This involves determining the process which is used
to generate a set of target values given a set of input variables.
Formally, given a set of input variables
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/54d304ef1fb8e71c763c911f3777da9f.svg?invert_in_darkmode" align=middle width=126.63618pt height=32.25585pt/> and a set
of target variables
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/0f64c6872baa716741745e0320de5584.svg?invert_in_darkmode" align=middle width=120.243585pt height=32.25585pt/>, we seek
to find the function <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/3827df7d20f06250c09fd68e242010f0.svg?invert_in_darkmode" align=middle width=32.580075pt height=24.6576pt/>, which was used to generate the
target variables from the given input. This is a complex task as there
is often an element of noise added to the function <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/3827df7d20f06250c09fd68e242010f0.svg?invert_in_darkmode" align=middle width=32.580075pt height=24.6576pt/>
before the target variable is produced. To keep things simple, we will
focus on data containing a 1D input variable <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.3951pt height=14.15535pt/> and a 1D output target
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.9361555pt height=20.22207pt/>. furthermore, we turn to a polynomial function, in order to
approximate our underling function <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/3827df7d20f06250c09fd68e242010f0.svg?invert_in_darkmode" align=middle width=32.580075pt height=24.6576pt/>. This is due to the
fact that most functions can be accurately approximated by a few terms
of their Taylor expansion. The parametric function used can therefore be
written as follows: <p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/8bc5f02117a875e7488860a3ee552edd.svg?invert_in_darkmode" align=middle width=134.64693pt height=77.022495pt/></p> Where <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode" align=middle width=17.73981pt height=22.46574pt/> is the order of the polynomial function and
plays a role in the maximum complexity of the function. We have also
defined <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/ae386aac36d6a996e53b1d1aca58f51f.svg?invert_in_darkmode" align=middle width=31.97502pt height=24.6576pt/> as <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/82b8d298061b6d06bc61b998e6a4d228.svg?invert_in_darkmode" align=middle width=117.743835pt height=35.54364pt/>. Further,
for convenience when handling multiple data-points, we define <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5e16cba094787c1a10e568c61c63a5fe.svg?invert_in_darkmode" align=middle width=11.872245pt height=22.46574pt/> as
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/26e9ed7922fd5b88d4bb52439761ace1.svg?invert_in_darkmode" align=middle width=134.422035pt height=32.25585pt/>. Here <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode" align=middle width=14.999985pt height=22.46574pt/>
denotes the number of data-points.\
We can now see that in order to approximate the given function
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/3827df7d20f06250c09fd68e242010f0.svg?invert_in_darkmode" align=middle width=32.580075pt height=24.6576pt/>, we must determine suitable values for the parameters
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5ddc1b22140b2658931d8962d8c90c33.svg?invert_in_darkmode" align=middle width=13.915605pt height=14.61207pt/>. In this exercise we explore the use of Bayesian and
classical methods to achieve this.

### Least Squares Approach

The least squares approach tries to minimise the squared error between
the target variables and the parametrised function. This error function
is defined as follows: <p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/7458c4e877e6bace99f5351682241330.svg?invert_in_darkmode" align=middle width=217.9287pt height=105.475095pt/></p> This is then minimised in close form by taking the
derivative with respect to W and setting this to zero.
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/24c02c0e97a558b056773701e94e4854.svg?invert_in_darkmode" align=middle width=293.9343pt height=47.80611pt/></p>

The result is given by [@christopher2016pattern] in equation 3.15 as:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/b9be2536e74946d8fcb4e91beb8a9fdd.svg?invert_in_darkmode" align=middle width=99.512325pt height=14.6503005pt/></p>

### Maximum Likelihood Approach

We now consider the likelihood of obtaining the target data from the
parametric function. For this we assume that the data has a Gaussian
distribution around the given function at any given input <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.3951pt height=14.15535pt/>. This is
therefore written as follows:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/f6880deb79d068fbcc6c8b6e7436072e.svg?invert_in_darkmode" align=middle width=281.91735pt height=47.80611pt/></p>
Where <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16565pt height=22.83138pt/> is the precision of the Gaussian distribution. Taking the
natural logarithm of this function we get:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5d0d043bb62c2489db1e55fac48cb3d2.svg?invert_in_darkmode" align=middle width=448.18125pt height=47.80611pt/></p> Minimising the log likelihood is identical to
minimising the sum of squares error. This can be seen in equation
\[eqn:E3:MLErr\], as the only term dependant on <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5ddc1b22140b2658931d8962d8c90c33.svg?invert_in_darkmode" align=middle width=13.915605pt height=14.61207pt/> is a scalar
multiple of the least squares error function. Due to this,
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/7c4f8bf0de31f2d40f899c43ea627433.svg?invert_in_darkmode" align=middle width=36.440745pt height=14.61207pt/> can be determined with the use of equation
\[eqn:E3:Wml\].

### Bayesian Approach

The Bayesian approach attempts to determine the probability of the
parameters <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5ddc1b22140b2658931d8962d8c90c33.svg?invert_in_darkmode" align=middle width=13.915605pt height=14.61207pt/> given the target variables <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/f40598ec49a99f9a93c399f7dacc6d3e.svg?invert_in_darkmode" align=middle width=7.3516245pt height=20.87415pt/>.
Assuming this takes a Gaussian form, we can model this probability as
follows:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5581b5873cff0870bc8a17e25e8eee5e.svg?invert_in_darkmode" align=middle width=177.22155pt height=16.438356pt/></p>
Where <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/538ab17daf80cfe5c8b95197e50542ec.svg?invert_in_darkmode" align=middle width=27.39957pt height=14.61207pt/> represents the mean of the weights and
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/cd4cab1968cc2c4db61ed56fdf4b4c08.svg?invert_in_darkmode" align=middle width=22.148445pt height=22.55715pt/> represents the variance. These can be determined in a
Bayesian approach by assuming an initial <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5048bfe7b0b6616ed02659c4e9048473.svg?invert_in_darkmode" align=middle width=22.30602pt height=14.61207pt/> and
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/283d31e54a22901cff1917c7cff2fdd8.svg?invert_in_darkmode" align=middle width=17.054895pt height=22.55715pt/>. Equations 3.50 and 3.51 from [@christopher2016pattern]
can then be used to update these parameters. This update step is given
as follows: <p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/42b9526624114ad1ebc1793853a3293c.svg?invert_in_darkmode" align=middle width=202.0326pt height=47.54739pt/></p>

It is common practice to assume a zero mean for <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/5048bfe7b0b6616ed02659c4e9048473.svg?invert_in_darkmode" align=middle width=22.30602pt height=14.61207pt/> and a
large variation for <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/283d31e54a22901cff1917c7cff2fdd8.svg?invert_in_darkmode" align=middle width=17.054895pt height=22.55715pt/> corresponding to <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/f4e3d6a50cdd502c97841d32e659bd14.svg?invert_in_darkmode" align=middle width=17.74542pt height=22.55715pt/>.
Here <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/d8471e559d932f20f66bec32f6002e08.svg?invert_in_darkmode" align=middle width=7.168986pt height=22.55715pt/> is known as the identity matrix.

### Results and Discussion of above methods {#sec:E3:ResDesc}

We now run the above algorithms on a dataset containing 10 points with
corresponding <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.3951pt height=14.15535pt/> and <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.9361555pt height=20.22207pt/> values. We will assume that the data is
generated in such a manner that <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/30ebd2e824d57be5a5cd5e0a50f3467f.svg?invert_in_darkmode" align=middle width=61.30707pt height=22.83138pt/> and
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/cef3f8eba4bbeb53ba2f3c8c332273d8.svg?invert_in_darkmode" align=middle width=94.011555pt height=27.77577pt/>. Furthermore, we will assume a zero
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

We now fit the graphs for <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/7a14b5b83ebce3bb4dcb52e3abd3a6d5.svg?invert_in_darkmode" align=middle width=47.876565pt height=22.46574pt/>. The results of the least squares and
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
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/4cdbfe2af76258268e4db1ce69d651e3.svg?invert_in_darkmode" align=middle width=389.01555pt height=47.80611pt/></p>

From the term <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/ea4c17c7a166f0c34c806cfe77fc5464.svg?invert_in_darkmode" align=middle width=48.705195pt height=27.65697pt/>, it is possible
to see that <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c02467c7ef1a7ba90aede84bfc1b5eac.svg?invert_in_darkmode" align=middle width=76.113015pt height=24.6576pt/> is
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

We now use Bayesian methods to determine the best model <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/b5eaea000e06d5cf2e882f8fdbc71e36.svg?invert_in_darkmode" align=middle width=19.74093pt height=22.46574pt/>
out of a set of models to explain the underling data <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/eaf85f2b753a4c7585def4cc7ecade43.svg?invert_in_darkmode" align=middle width=13.137135pt height=22.46574pt/>. For
this we need to evaluate <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/376d58310d3ad1980cb25008afe8788a.svg?invert_in_darkmode" align=middle width=63.972975pt height=24.6576pt/>. For this, we
can use bays rule which states:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/2c7dd0622c930639630251abdde63c19.svg?invert_in_darkmode" align=middle width=198.1056pt height=38.834895pt/></p> If we assume that the prior probability
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/2ca083183cb3adaa98b813eb71984394.svg?invert_in_darkmode" align=middle width=46.26963pt height=24.6576pt/> is constant over all models. Then we can simplify
equation \[eqn:E3:MgD\] to
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/51108e1e5cd45431e22d8c7baf2235cc.svg?invert_in_darkmode" align=middle width=212.0547pt height=16.438356pt/></p> Therefore, it is equivalent to work out the
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/4ffe7dbb63e9eb815f60501c70d0d5a9.svg?invert_in_darkmode" align=middle width=63.972975pt height=24.6576pt/> and normalise over all the models.
Therefore, when comparing a list of polynomial functions, we can use
<img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/6f93eae730d5f783e91a0c498fdb4d49.svg?invert_in_darkmode" align=middle width=82.243095pt height=24.6576pt/> to determine the best model for
the data. This is known as the evidence function. The formula required
to calculate this is given by [@christopher2016pattern] in equation
3.78. This states that:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c514657fd6d9bb9ccb9f9e99315591aa.svg?invert_in_darkmode" align=middle width=390.84045pt height=44.0187pt/></p>
Where we can use equation 3.85 from [@christopher2016pattern], which
states:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/e13b789a5784e0fde7cef5586d1a45b6.svg?invert_in_darkmode" align=middle width=391.4889pt height=36.53001pt/></p>
In order to compute <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/6f93eae730d5f783e91a0c498fdb4d49.svg?invert_in_darkmode" align=middle width=82.243095pt height=24.6576pt/>, we also
require the following: <p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/59cdcfbc48dbfd531138a0e65c53e95a.svg?invert_in_darkmode" align=middle width=274.43955pt height=58.23411pt/></p> Using a new
dataset containing 80 samples produced in a similar fashion to the
dataset used in section \[sec:E3:ResDesc\], we can evaluate the evidence
as given above. This is done for <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/6a020d5efdf9b539c96c7e6c33cb222a.svg?invert_in_darkmode" align=middle width=47.876565pt height=22.46574pt/> up until <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/7a14b5b83ebce3bb4dcb52e3abd3a6d5.svg?invert_in_darkmode" align=middle width=47.876565pt height=22.46574pt/>. The results of
this are shown in figure \[fig:E3:Evi\].

![Plot of the model evidence for various values of
M.[]{data-label="fig:E3:Evi"}](Figs/Q3/Q3P6Evidance){width="60.00000%"}

We can see from this that the best for to the data relates to <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/caebfe10e001be633bb0854921be2243.svg?invert_in_darkmode" align=middle width=47.876565pt height=22.46574pt/>. To
justify this result, we can turn to the Taylor expansion of a sine
function. This is given by:
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/52b4ec4feaa59aa90a9f5b537ae52941.svg?invert_in_darkmode" align=middle width=232.93545pt height=35.777445pt/></p> This is an
odd function and hence even powers of <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.3951pt height=14.15535pt/> do not contribute to the final
form of the function. Furthermore, the factorial in the denominator of
each term, means that the contribution of each term diminishes quickly.
These observations can be seen in figure \[fig:E3:Evi\] as <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/caebfe10e001be633bb0854921be2243.svg?invert_in_darkmode" align=middle width=47.876565pt height=22.46574pt/> is a
clear maximum followed by a sudden and sharp drop in preceding terms.
The plot corresponding to the most likely model is given in figure
\[fig:E3:m3:DTA2\].\

![Plot of the fitted curve for
M=3[]{data-label="fig:E3:m3:DTA2"}](Figs/Q3/Q3P6_Order3){width="60.00000%"}

### Bayesian Model Averaging

We now average all the models tested in section \[sec:E3:ModComp\]. For
this, we can take a weighted sum over the model space. This is given by
equation 3.67 from [@christopher2016pattern].
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/d7fae5b4877e6d093e19219af0f4b950.svg?invert_in_darkmode" align=middle width=297.06105pt height=47.988765pt/></p>
We can determine <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c23bbf51615b2c84bd4fbd24ff077f86.svg?invert_in_darkmode" align=middle width=64.64172pt height=24.6576pt/> by normalising over the
evidence function due to equation \[eqn:E3:EVeqProb\]. We can then use
equations \[eq:E3:muave\] and \[eq:E3:sigmaave\], provided by
[@trailovic2002variance], to determine the mean and standard deviation
of the weights. Note that these equations only estimate the mean and
variance of the distribution. This is due to the fact that a mixture
distribution will most likely be multi-modal and contain more that one
local maximum.

<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/d476398858621118924527597ce86556.svg?invert_in_darkmode" align=middle width=347.46855pt height=105.84057pt/></p>

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

In order to determine <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> and <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16565pt height=22.83138pt/>, we first need to assume an
initial <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> and <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16565pt height=22.83138pt/>. We then compute <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/538ab17daf80cfe5c8b95197e50542ec.svg?invert_in_darkmode" align=middle width=27.39957pt height=14.61207pt/> using
equation \[eqn:E3:mn\] with the initial guess of <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> and <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16565pt height=22.83138pt/>.
We then need to compute the following two values: <p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/485cf471bdce549066e131d5b41d058f.svg?invert_in_darkmode" align=middle width=247.0116pt height=89.01552pt/></p>
This is then used to compute the new parameters <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.5765pt height=14.15535pt/> and <img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/8217ed3c32a785f0b5aad4055f432ad8.svg?invert_in_darkmode" align=middle width=10.16565pt height=22.83138pt/>
using equations 3.98 and 3.99 from [@christopher2016pattern].
<p align="center"><img src="https://rawgit.com/ub	git@github.com:sjnarmstrong/basic-regression-methods/master/svgs/749f03fa6a03a82038dec99ed500e5dd.svg?invert_in_darkmode" align=middle width=106.468725pt height=82.053345pt/></p> This is then repeated
until convergence or until a maximum number or iterations is reached. It
is important to note that this method is only valid when the number of
data points is greatly larger than the order of the polynomial function.
If this is not the case, one must employ a more complicated procedure.

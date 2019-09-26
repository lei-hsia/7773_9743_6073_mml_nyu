Even if the matrix is not invertible, a solution can still exist for an equation involving this matrix;

http://www.atteson.com/Markov/linearalgebra.html

#### Eigenvalues && eigenvectors

Not every matrix is diagolizable, but every matrix can be put into a ```Jordan normal form```;

这个例子要看: http://www.atteson.com/Markov/stationary.html#gcd2

```
every random variable DOES HAVE A CUMULATIVE DENSITY FUNCTION, but not every random variable has a density distribution function, not every random variable has a point mass function.
```

a random variable is a function from Probability Space to R.

---

Just over the "Some Classical Limit Theorems" part: 

> 改正: αX的density函数是f<sub>α</sub>(x) = (1/α) * f(X/α)
  
#### The Central Limit Theorem部分: 
1. limit(n->infinity) SIGMA(X - E[X]) / n : 得到的是```law of large numbers```; 因为除以n, 所以有很明显的dampen; 波动的极差很小了;
2. limit(n->infinity) SIGMA(X - E[X]) / √n : 得到的是```central limit theorem```; 因为√n, 并没有除以n那样的dampen, 此时看起来像是stationary, 对应的是central limit theorem;
3. limit(n->infinity) SIGMA(X - E[X]): 什么都不除, 对应的是```random walk```

#### MLE: 
分母除以```n```的是 max likelihood estimator, 而不是除以```n-1```

---

#### Maximum Likelihood Estimation (MLE)

对于大多数分布, MLE都是 sample mean: e.g. normal distribution: mean是最多的分布变量; binomial: MLE对应的是head的次数; 
```但是, 并不是所有的MLE都是sample mean: e.g. Cauchy distribution：根本都没有mean; or ```

```Example: Laplace Distribution: 也是exp(...), 只是正态分布的指数没有平方，而是一次方; 这个的MLE算出来是sample median, 不是sample mean```

>> 讲到了MLE for IID random variable;

#### MLE for Markov Chain: 
当MC的某个状态是transient的时候, 并不是recurrent, 不能用大数定律估计, 此时不能用MLE估计
this path happens with positive probability, the estimate is inconsistent for a single sample path. 

1st-order MC: 后面一期的状态只取决于当前的状态; 
2nd-order MC: 后面一期的状取决于它自己前面两期的状态, e.g. current && previous states

---

Model Order Selection && Hypothesis Testing 

```log likelihood ratio test```






1. Markov chains, &&, martingales: 2 main sets of dependencies process;

2. MC: "the future is independent of the past given the present"

3. memoryless:
  - ```exponential distribution``` is the only continuous memoryless distribution;
  - ```geometric distribution``` is the discrete version;

4. ```Multi-step Transition Probabilities```: 上面的三行话没懂

5. Example: Simple Model of Prepayments and Defaults

6. (below the rating graph): Stationary Distribution: it's also reflected in the rating graph,
where graphs seem to reach a plateau, or very flat. This is when it's converging to
stationary distributions.

7. ⚠️stationary distribution中的从 两个summation到 1个summation的过渡

8. Any homogeneous MC has a stationary distribution, and it's unique;

9. **the stationary distribution is unique**, the distribution of the random 
variables need not converge to the stationary distribution

10. We can solve for the stationary distribution by solving a system of linear
equations. Similar to absorption probabilities, there is also a method of deriving 
a stationary distribution of a Markov chain by using graph theory which 
we present later after more formally defining some terms.

11. tree: maximum cycle-free graph; or: minumum connected graph

12. A state ```s``` of a Markov chain is said to be recurrent if it occurs 
again, given that it occurs once, with probability ```1```.

13. If you can go to any state from other states, i.e. you can traversal all nodes 
in this graph, then this graph is ```irreducible```.

14. Context: "2 regime" represent 2 economic regimes: e.g.
  - 1: depression;
  - 2: normal;
  
15. 3 regimes: 
  - 1: depressed;
  - 2. intermediate;
  - 3. buoyant:
  - attention that regimes can't be determined clearly: **hidden MC**

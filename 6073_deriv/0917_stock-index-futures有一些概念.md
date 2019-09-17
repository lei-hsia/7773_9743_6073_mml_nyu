1. Convergence: of futures;
Hull's hedging position is wrong:

**actually, it's pretty simple to understand:**
 ```
 Note: 
  1. P0: portfolio value at t = 0;
  2. F0, Ft: futures value at t = 0 && t = exp;
  3. I0, It: underlying asset value;
  
 Hull's hedging position := P0 / (multiplier * F0);
 what should be          := P0 / (multiplier * I0);
 ```
 hedging position: **using the index**: ```P0 / (multiplier * I0);```, not using ```F0``` to be the denominator;
 **because F0 would converge to It/Ft, which results in the wrong position number**
 
 2. ```bona fide hedging```: 
 
  a. bona fide (Latin): good faith: means when you have the inventory and wants to hedge it, e.g. A farmer in the midwest wants 
 to hedge against a falling corn price: bona fides hedging;
  b. when you don't have the corresponding inventory in stock, i.e. you are essentially speculating against the falling price,
  then you put the financial market under risks, so you have to pay a margin, at a good amount.
  
3. Different kinds of hedging: 

  a. ```long hedge```: a LONG hedge refers to the futures position entered for the porpose of price stability for a purchase,
  i.e. long hedge is entering a long in the futures contract.
  
  b. ```short hedge```: SHORT hedge: sell safe: entering a short position;
  
  c. ```ß hedge```: ß increases volatility: hedging increases by the same level;
  
4. [optimal hedging](https://breakingdownfinance.com/finance-topics/risk-management/market-risk/optimal-hedge-ratio/)


> ### This concludes **stock index futures**

1. A still mroe realistic model of prepayment: 
  No 60 to 30: based on real life experience.
 
2. L, L+1, ... I-1, U: 
  - once going down, going down forever, because it can't go in the reverse way, otherwise
  it's not a DAG;
  - similarly, once going up, going up forever. 
 
3. In formula (2): 
  when ```u==d```, then ```q= (S0-L)/(U-L)```
  
  this can also be solved via martingale: ```E[Pt, hitting U/L] = X1 = S0```; 
  then you can get the same result.
  

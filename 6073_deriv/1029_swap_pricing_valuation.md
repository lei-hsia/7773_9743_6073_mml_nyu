#### pricing a swap;
1 year reset swap: base = 365;

| k | FRA | Rate  |  days | 
| - |:-----:| -----:| : -- : |
| 1 | 0*3 | 6.50% |  92 |
| 2 | 3*6 |  5.25 | 91 |
| 3 | 6*9 |  6.00 | 90 |
| 4 | 9*12| 6.25 | 92 |


#### valuing a swap:
2 year semi-annual reset 8% swap on $100 million; 6-mo LIBOR = 10.20% 3 months ago;
 
V(t) = B_fix - B_float; 

B_fix = 99.09; B_float = 102.59; ==> value = -3.50

#### rest seen in notes

Swap := a unhedgeable part + a set of FRAs: because if you see from the above valuing swap part, the hedgeable part is in the form of ```Notation * (r_fix - r_float)^time-duration```, which is in the same form as the payments of FRAs.


#### Off-market swap: 

1. 3-party diagrams: a company XYZ, a note-provider predefined w. XYZ, and a dealer with which XYZ sets a swap. 
2. XYZ w. note: predefined swap w. rate ```S'```;
3. XYZ w. dealer: current swap rate ```S```; 
4. ```S``` and ```S'``` have a difference: 
5. a fee, is paid from dealer to XYZ if ```S``` > ```S'```;
6. fee = (S - S')/2 * sigma_discount-factor;
7. direction of fee is dependent on which rate is higher

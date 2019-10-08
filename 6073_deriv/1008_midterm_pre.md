Content: 
1. T-bond futures, cheapest-to-deliver;
2. FRAs;
3. hedging;
4. 2: Eurodollar futures, slide_one, challenge;
5. All calculations: discrete, not continuous;
6. Show the work, put down the calculations; 
7. Read all instructions carefully, read twice before writing;

When, where:

1. ```9AM-noon; RGSH 325```

---

Achtung: 
1. FRAs && Swaps;
2. FRN: Floating-rate note;
3. Macaulay durations, modified durations;
4. **Valuation curves**:
```
Normal problem: 
1. Forward curve
2. Zero-coupon curve
3. Swap curve
4. Par curve / current curve
```

Bootstrapping: (From English 17-th century English fox hunting).

This is the method we adopt to get all the zero rate, finally getting the zero curve from the 
par yield we have. 

Normal: **(recall from RBC interview):**
```
Bootstrapping to get the forward interest rate: 

Case: Suppose the current yield for 1 year is 10%, for 2-year is 12%, based on bootstrapping 
method to calculate the forward interest rate between [1y, 2y], it's easy to get ~14%. Now 
suppose both of them increase by 1%, is the forward rate going to increase by more than 1%, 
or less than 1%, or equal?

Answer: More than 1%. 
Because current yield to forward interest rate, is like average rate to marginal rate, and when
average of longer term (2-y) increase by 1%, the marginal must increase by more to keep up with 
the longer term increase.
```

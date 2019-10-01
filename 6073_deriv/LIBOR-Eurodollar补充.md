后来从 [Investopedia](https://www.investopedia.com/articles/active-trading/012214/introduction-trading-eurodollar-futures.asp) 上面看到的: 

The underlying instrument in eurodollar futures is an eurodollar time deposit, having 
a principal value of $1 million w. a 3-month maturity.

LIBOR && Eurodollars:

The price of eurodollar futures reflects the interest rate offered on USD-denominated deposits
held in banks outside US. More specifically, the price reflects the market gauge of the 3-month
USD LIBOR interest rate anticipated on the settlement date of the contract. LIBOR is a benchmark
for short-term interest rates at which banks can borrow funds in the London interbank market.
Eurodollar futures are a LIBOR-based derivative, reflecting the LIBOR for a 3-month $1 million
offshore deposit.

所以eurodollar futures 还是归为 interest rate derivatives.
今天(2019.10.01)学的两个 利率期货, ED futures 和FRA, 都是流动性非常高的利率期货, 
而且 eurodollar让我想起来我大学时代做的 GBP/USD FX交易和Brexit之前的short sterling

#### 下面这个例子非常重要
```
Eurodollar futures prices are expressed numerically using 100 minus the implied 3-month
USD LIBOR interest rate. In this way, an ED futures price of $96.00 reflects an implied
settlement interest rate of 4%.

e.g. if an investor buys 1 ED futures contract at $96.00 and the price rises to $96.02,
this corresponds to a lower implied settlement of LIBOR at 3.98%. Since the buyer buys 
at a lower price, he makes a profit. To be precise, he makes $50, because: 


1 basis point, 0.01%, == $25 per contract. 
This is because $1 million * 1 basis change(10^(-4)) = $100, corresponds to 1 year;
LIBOR corresponds to 3-month: 3/12 * 100 = $25.

Then $96.00 -> $96.02, a move of 0.02 equals a change of $50 per contract.
```

#### Hedging with Eurodollar Futures





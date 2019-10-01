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

所以eurodollar futures 还是归为 interest rate derivatives. <br>
今天(2019.10.01)学的两个 利率期货, ED futures 和FRA, 都是流动性非常高的利率期货, <br>
而且 eurodollar让我想起来我大学时代做的 GBP / USD通貨ペア外国為替取引 (GBP/USD FX交易) 和Brexit之前的short sterling; <br>
虽然说，实际上FX上的英美货币对交易，做空sterling，还有这个ED futures，这三个东西根本就是完全不相关的东西 ... ... GG <br>
#### そこには3つの概念が全く無関係です

---

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
---

#### Hedging with Eurodollar Futures

Eurodollar futures provide an effective means for companies and banks to secure an interest rate for money it plans to borrow or lend in the future. The eurodollar contract is used to hedge against yield curve changes over multiple years into the future.

For example: Say a company knows in September that it will need to borrow $8 million in December to make a purchase. Each eurodollar futures contract: $1 million time deposit with a 3-month maturity. The company can **hedge against an adverse move in interest rates** during that 3-month period by **short selling** 8 December eurodollar futures contracts, representing the $8 million needed for the purchase.

理解这里为什么是short selling来对冲: 因为3个月之后要borrow,所以风险是3个月的利率上升，这样还钱更多; <br>
也就是说，是在3个月的期望利率会上升的这个假设下，才用利率期货锁定现在的利率进行对冲的, 如果期望利率会下降，<br>
根本就不用进行对冲. 

从下面两个方面理解: 

1. 直接从futures的角度: 因为futures的价值和interest rate是反过来的, 一般都是看seller／short futures头寸, 因为总是担心LIBOR上升的风险，如果LIBOR上升，futures的会下降, 所以现在锁定现在的价格卖出，是会有payoff的，这也对应了老师讲的 LIBOR increase, seller profit increase.

```
逻辑: 3个月之后要borrow --> 担心利率会上升 --> 利率如果上升 == 期货价格会下降 --> 那么要对冲，就以现在的高期货价格卖出现在的期货,锁定高价带来的利润 --> short futures now
```

The price of eurodollar futures reflects the anticipated London Interbank Offered Rate (LIBOR) at the time of settlement, in this case, December. By short selling the December contract, the company profits from upward movement in interest rates, reflected in correspondingly lower December eurodollar futures prices.

Let’s assume that on Sept. 1, the December eurodollar futures contract price was exactly $96.00, implying an interest rate of 4.0%, and that at the expiry in December, the final closing price is $95.00, reflecting a higher interest rate of 5.0%. If the company had sold eight December eurodollar contracts at $96.00 in September, it would have profited by 100 basis points (100 x $25 = $2,500) on eight contracts, equaling $20,000 ($2,500 x 8) when it covered the short position.

In this way, the company was able to offset the rise in interest rates, effectively locking in the anticipated LIBOR for December as it was reflected in the price of the December eurodollar contract at the time it made the short sale in September.

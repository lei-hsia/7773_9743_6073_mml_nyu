1. 记住那个蓝色的箭头➡️: 对应的是delivery month那个月; 有5个不同的时间: 
```
  1. First Position Day;
  2. Last Trading Day;
  3. Intention day;
  4. Notice day;
  5. Delivery day;
```
其中, ```3,4,5```是可以在这个月, which is roughly constrainted by ```1``` and ```2```, 像sliding window那样自由滑动的;
只是这个sliding window的size是固定的;

First Position day: google to see definition;

2. carry := Coupon Income(CI) - Financing Cost(FC)

  - ```Repo rate(interest rate) < Coupon rate: positive carry: futures price:下降;```
  分析这个首先注意两个rate分别对应哪一方: futures的两方: short方现在持有T bond, 在futures规定的将来某个时间要卖/deliver bond,
  long方现在有钱,将来某个时间要买; 所以coupon rate是跟short方相关的, repo rate(重新financing)是跟long方相关的; 
  - 用极端法分析: 现在假设interest rate是0, 
 
 >> 怪了, 感觉这样分析出来的futures价格的变动反过来了

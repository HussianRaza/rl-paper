# OPHR -- Financial Topics Explained Simply

---

## Part 1: Options (The Foundation)

### What is an Option?

An option is a **contract** that gives you the **right** (but not obligation) to buy or sell something at a fixed price in the future.

**Real-life analogy:** You see a house listed for $300,000. You pay the owner $5,000 for the RIGHT to buy it at $300,000 anytime in the next 3 months. If the house price goes to $400,000, you exercise your right, buy at $300,000, and you've made $95,000 profit ($100K gain - $5K premium). If the price drops to $250,000, you just walk away and lose the $5,000 premium.

### Call vs Put

```
CALL OPTION = Right to BUY
  "I can buy Bitcoin at $50,000 before March 30"
  Profitable when price GOES UP

PUT OPTION = Right to SELL  
  "I can sell Bitcoin at $50,000 before March 30"
  Profitable when price GOES DOWN
```

### Key Option Terms

```
UNDERLYING ASSET  = What the option is based on (Bitcoin, Ethereum)
STRIKE PRICE (K)  = The fixed buy/sell price in the contract
EXPIRATION (T)    = When the contract expires
PREMIUM           = Price you pay to buy the option
```

### Option Payoff at Expiration

```
Call payoff = max(Price - Strike, 0)
  BTC at $60K, Strike $50K: payoff = $10K
  BTC at $40K, Strike $50K: payoff = $0 (worthless)

Put payoff = max(Strike - Price, 0)
  BTC at $40K, Strike $50K: payoff = $10K
  BTC at $60K, Strike $50K: payoff = $0 (worthless)
```

### At The Money (ATM)

```
ATM = Strike price equals current market price

If BTC is at $50,000:
  $50,000 strike call = ATM (at the money)
  $55,000 strike call = OTM (out of the money) -- cheaper, less likely to pay off
  $45,000 strike call = ITM (in the money) -- more expensive, already has value
```

OPHR only trades ATM options because they have the highest sensitivity to volatility (highest Gamma and Vega).

---

## Part 2: The Straddle (OPHR's Trading Instrument)

### What is a Straddle?

A **straddle** = Buy a Call AND a Put at the SAME strike price and expiration.

```
Example: BTC at $50,000
  Buy 1 Call at $50K strike  (costs $2,000 premium)
  Buy 1 Put at $50K strike   (costs $2,000 premium)
  Total cost: $4,000

Outcomes:
  BTC goes to $60,000: Call worth $10K, Put worth $0  -> Profit = $10K - $4K = $6K
  BTC drops to $40,000: Call worth $0, Put worth $10K -> Profit = $10K - $4K = $6K
  BTC stays at $50,000: Call worth $0, Put worth $0   -> Loss = -$4K (total premium)
```

**Key insight:** A straddle profits from BIG MOVES in EITHER direction. It loses money when the price stays still.

### Long Straddle vs Short Straddle

```
LONG STRADDLE (buying both call and put):
  You PAY the premium
  You PROFIT from big moves
  You LOSE from time decay (theta)
  Risk: limited to the premium paid
  Analogy: Buying earthquake insurance -- you profit from the disaster

SHORT STRADDLE (selling both call and put):
  You RECEIVE the premium
  You PROFIT from calm markets (collect premium, options expire worthless)
  You LOSE from big moves
  Risk: UNLIMITED (price could move infinitely)
  Analogy: Selling earthquake insurance -- you earn premium but face catastrophic payouts
```

OPHR decides when to be in a long straddle (expecting chaos) or short straddle (expecting calm).

---

## Part 3: Volatility (The Core Concept)

### What is Volatility?

Volatility measures **how much a price moves around**. It's expressed as an annualized percentage.

```
Low volatility (10%):  BTC might move ±$5,000/year from $50K = range of $45K-$55K
High volatility (80%): BTC might move ±$40,000/year from $50K = range of $10K-$90K
```

### Implied Volatility (IV)

IV is the volatility **the market expects**. It's embedded in option prices.

```
If an ATM call option costs $2,000 and BTC is at $50,000:
  "The market thinks BTC will move enough to justify a $2,000 premium"
  This implied expectation = IV

Higher IV = more expensive options (market expects bigger moves)
Lower IV  = cheaper options (market expects calm)
```

**How to think about it:** IV is the market's "fear gauge." When traders are scared of big moves, they bid up option prices, which raises IV.

### Realized Volatility (RV)

RV is the volatility that **actually happened**. Measured after the fact using real price data.

```
This week:
  Monday:    BTC $50,000
  Tuesday:   BTC $52,000 (+4%)
  Wednesday: BTC $48,000 (-7.7%)
  Thursday:  BTC $51,000 (+6.25%)
  Friday:    BTC $50,500 (-0.98%)
  
  These daily moves are large -> RV is high
  
vs last week:
  Monday:    BTC $50,000
  Tuesday:   BTC $50,200 (+0.4%)
  Wednesday: BTC $50,100 (-0.2%)
  Thursday:  BTC $50,300 (+0.4%)
  Friday:    BTC $50,250 (-0.1%)
  
  These daily moves are tiny -> RV is low
```

### The IV-RV Gap (Where the Money Is)

```
CASE 1: IV = 50%, RV turns out to be 70%
  Market expected moderate moves, but BIG moves happened
  Long straddle holders profit (bought "cheap" insurance, got big payoff)
  Short straddle holders lose (sold "cheap" insurance, face big claims)

CASE 2: IV = 50%, RV turns out to be 30%
  Market expected moderate moves, but market was CALM
  Long straddle holders lose (overpaid for insurance they didn't need)
  Short straddle holders profit (collected premium, paid out nothing)
```

**This is what OPHR tries to exploit.** The OP-Agent predicts whether RV will be higher or lower than IV and trades accordingly.

### The Volatility Risk Premium

IV is USUALLY higher than RV. Why?

```
Option sellers demand extra compensation for bearing tail risk
  (the risk of a 20% crash in one day)

So on average:
  IV = 50%
  RV = 40%
  
  This means selling options is profitable MOST of the time
  But occasionally, a crash happens (RV = 200%) and wipes out months of profits
```

This is why "always sell options" doesn't work long-term (see paper results: Short strategy only +2.9% on BTC with terrible risk metrics).

---

## Part 4: The Greeks (Risk Measures)

### Why Are They Called Greeks?

They're named after Greek letters: Delta, Gamma, Theta, Vega (Vega isn't actually a Greek letter but the name stuck).

### Delta -- Price Sensitivity

```
Delta = How much does my portfolio value change when BTC moves $1?

Delta = +0.5: If BTC goes up $100, my portfolio gains $50
Delta = -0.3: If BTC goes up $100, my portfolio LOSES $30
Delta = 0:    BTC price movement doesn't affect my portfolio (delta-neutral)
```

**For a long ATM straddle:**
```
Call delta = +0.5 (gains when BTC goes up)
Put delta  = -0.5 (gains when BTC goes down)
Straddle delta = +0.5 + (-0.5) = 0 (initially delta-neutral!)
```

But delta CHANGES as price moves, which is where Gamma comes in.

### Gamma -- How Fast Delta Changes

```
Gamma = How much does Delta change when BTC moves $1?

High Gamma: Delta changes rapidly with price
Low Gamma:  Delta barely changes

For a long straddle:
  BTC goes up $1000 -> delta becomes +0.1 (now exposed to upside)
  BTC goes up another $1000 -> delta becomes +0.2 (even more exposed)
  
  Gamma makes your straddle "lean into" whichever direction the market moves
  This is the source of profit from big moves!
```

**Gamma is the reason straddles are profitable during big moves:**
- Price goes up -> delta turns positive -> you're effectively long -> price keeps going up -> you profit
- Price goes down -> delta turns negative -> you're effectively short -> price keeps going down -> you profit

```
Long straddle:  Gamma > 0  (benefits from big moves in either direction)
Short straddle: Gamma < 0  (harmed by big moves -- this is the tail risk)
```

### Theta -- Time Decay

```
Theta = How much value does my option lose PER DAY from time passing?

Theta = -0.005: My straddle loses $0.005 BTC per hour just by existing
```

**Why does this happen?** An option's value comes from the POSSIBILITY of a big move. As time passes, there's less time for that big move to happen, so the option becomes less valuable.

```
Day 1:  Straddle worth $4,000 (30 days of possibility)
Day 15: Straddle worth $2,800 (15 days of possibility -- less chance of big move)
Day 29: Straddle worth $500   (1 day left -- almost no chance)
Day 30: Straddle worth $0     (expired, no more possibility)

This decay accelerates near expiration (the "$500 to $0" drop happens fast)
```

**For OPHR:**
```
Long straddle:  Theta < 0 (you PAY time decay every hour -- this is your "rent")
Short straddle: Theta > 0 (you COLLECT time decay every hour -- this is your "income")
```

### Vega -- Volatility Sensitivity

```
Vega = How much does my portfolio change when IV changes by 1%?

Vega = +0.10: If IV goes from 50% to 51%, my portfolio gains $0.10 BTC
```

**Why this matters:**
```
Long straddle:  Vega > 0 (you profit if IV increases -- market gets more scared)
Short straddle: Vega < 0 (you profit if IV decreases -- market calms down)
```

If you buy a straddle and then a market panic hits, IV spikes, and your straddle becomes much more valuable even before BTC actually moves much. This is the "Vega profit" shown in the paper's PnL decomposition.

### Greeks Summary Table

```
               Long Straddle        Short Straddle
               (Buy options)        (Sell options)
Delta          ~0 (neutral)         ~0 (neutral)
Gamma          Positive             Negative
               (profit from moves)  (hurt by moves)
Theta          Negative             Positive
               (pay time decay)     (earn time decay)
Vega           Positive             Negative
               (profit if IV rises) (profit if IV falls)

PnL = Gamma effect + Theta effect + Vega effect + Delta hedging effect
```

---

## Part 5: Delta Hedging (Risk Management)

### What is Delta Hedging?

Your straddle starts delta-neutral (delta = 0), but as BTC moves, delta changes (because of gamma). Delta hedging means **trading BTC futures to bring delta back to zero**.

```
Step-by-step example:

Hour 0: Buy straddle, portfolio delta = 0
        BTC = $50,000

Hour 1: BTC rises to $51,000
        Portfolio delta = +0.05 (now exposed to BTC going up)
        HEDGE: Sell 0.05 BTC futures -> delta back to 0

Hour 2: BTC drops to $49,000
        Portfolio delta = -0.08
        HEDGE: Buy 0.08 BTC futures -> delta back to 0

Hour 3: BTC drops further to $48,000
        Portfolio delta = -0.12
        HEDGE: Buy 0.12 BTC futures -> delta back to 0
```

### Why Hedge?

Hedging isolates the **volatility bet** from the **direction bet**.

Without hedging: Your straddle could lose money even if RV > IV, because the DIRECTION of the move matters.

With hedging: You only care about HOW MUCH the price moves, not WHICH DIRECTION.

### The Hedging Dilemma (Why Strategy Selection Matters)

```
OSCILLATING MARKET: BTC goes 50K -> 48K -> 52K -> 50K

  Aggressive hedging (threshold 0.05):
    Sell at 52K, buy at 48K, sell at 52K... 
    You're buying low and selling high = PROFIT from hedging!
  
  No hedging:
    Price ends where it started, your straddle is flat
    You missed the hedging profits

TRENDING MARKET: BTC goes 50K -> 55K -> 60K -> 65K

  Aggressive hedging (threshold 0.05):
    Sell at 51K (hedge), sell more at 52K, sell more at 53K...
    You keep selling into a rally = LOSS from hedging
  
  No hedging:
    Your straddle benefits from the large 30% move
    You captured the full gamma profit
```

**This is exactly why the HR-Agent exists.** It needs to figure out:
- Oscillating market? -> Use aggressive hedger (capture profits)
- Trending market? -> Use conservative hedger (let gains run)

### Types of Hedgers in OPHR

**Delta-Threshold Hedgers:**
```
Monitor portfolio delta, hedge when |delta| exceeds threshold

Hedger(threshold=0.05): Very aggressive
  Hedges when delta moves just 0.05 away from zero
  Good for: oscillating markets (locks in small profits frequently)
  Bad for: trending markets (sells into momentum repeatedly)

Hedger(threshold=0.30): Conservative
  Only hedges when delta is quite large (0.30)
  Good for: trending markets (lets profits run longer)
  Bad for: oscillating markets (misses profit-locking opportunities)

Hedger(threshold=3.0): Basically never hedges
  Extreme risk-taker -- lets the straddle run unhedged
```

**Price-Move Hedgers:**
```
Instead of watching delta, watches the PRICE itself

Hedger(price_threshold=0.01): Hedges when BTC moves 1%
Hedger(price_threshold=0.03): Hedges when BTC moves 3%
```

---

## Part 6: Perpetual Futures (The Hedging Instrument)

### What is a Perpetual Future?

A perpetual future (perp) is a contract that tracks the price of BTC **without an expiration date**.

```
Regular futures: "Buy BTC at $50K on March 30" (expires)
Perpetual futures: "Long/short BTC at current price, forever until you close"
```

Perps are used for hedging because:
- No expiration (no need to roll positions)
- Very liquid (easy to trade large amounts)
- Trade 24/7 (matches crypto options)

### Funding Rate

Perps have a **funding rate** -- a periodic payment between long and short holders to keep the perp price close to the spot price.

```
If funding rate = +0.01%:
  Longs pay shorts 0.01% every 8 hours
  "The market is bullish, so longs pay a premium to stay long"

If funding rate = -0.01%:
  Shorts pay longs 0.01% every 8 hours
```

OPHR's environment includes funding rate costs in the PnL calculation, making it realistic.

### Inverse Contracts (How Deribit Works)

On Deribit, BTC options and perps are **inverse contracts** -- they are denominated in BTC, not USD.

```
Regular contract: You profit/lose in USD
Inverse contract: You profit/lose in BTC

Example:
  Long 1 BTC perp at $50,000
  BTC goes to $55,000
  Profit = 1 * (1/$50,000 - 1/$55,000) * $50,000 = 0.0909 BTC

This is why the code uses Decimal math and BTC-denominated accounting.
Initial capital in OPHR = 10 BTC (not $500,000)
```

---

## Part 7: Portfolio Margin (Deribit's Risk System)

The exchange doesn't let you take unlimited risk. It calculates how much margin (collateral) you need:

### Risk Matrix

```
The exchange simulates your portfolio under 27 scenarios:
  9 price moves:  -16%, -12%, -8%, -4%, 0%, +4%, +8%, +12%, +16%
  x 3 vol shocks: 75% of current IV, 100%, 150%

For EACH scenario:
  "If BTC dropped 16% and IV went to 150%, how much would this portfolio lose?"

Worst-case loss across all 27 = base margin requirement
```

### Delta Shock and Roll Shock

Additional margin for:
- **Delta shock:** Extra risk from concentrated directional bets
- **Roll shock:** Risk from options approaching expiration

```
Total Initial Margin = Risk Matrix worst loss + Delta Shock + Roll Shock

If your margin exceeds your available capital:
  The trade is REJECTED by the environment (just like a real exchange)
```

This is implemented in `base_env.py` and is one of the reasons the simulation is considered realistic.

---

## Part 8: Transaction Costs

### Fee Structure (Deribit)

```
Options:      0.03% per contract (capped at 12.5% of option price)
Perpetuals:   0.05% of notional value
Combo trades: Second leg free (straddle = call + put, put is free)
```

### Why Costs Matter

```
Without costs: Strategy earns 40%
With costs:    Strategy earns 33% (costs consumed 7% of PnL)

Many academic papers ignore costs, making their results unrealistically good.
OPHR includes costs and still profits:
  BTC: costs = 9.36% of PnL (strategy is robust)
  ETH: costs = 5.75% of PnL (even better)
```

---

## Part 9: The Gamma-Theta Relationship (The Key Equation)

This is the most important financial equation in the paper:

```
PnL of delta-hedged straddle = integral from 0 to T of:
  (S_t^2 / 2) * Gamma * (RV^2 - IV^2) dt
```

**In plain English:**

```
Your profit = Sum over all time of:
  [How explosive your position is (Gamma)]
  x [How much ACTUAL moves (RV) exceeded EXPECTED moves (IV)]

If RV > IV throughout the holding period:
  Every term is positive -> you profit from long gamma

If RV < IV throughout the holding period:
  Every term is negative -> you lose on long gamma (but profit on short gamma)
```

**The intuition:**

```
Long straddle (positive Gamma):
  Each hour, you earn: Gamma * (actual price move^2 - expected price move^2)
  Each hour, you pay: Theta (time decay)
  
  Gamma earnings > Theta cost  when  RV > IV  --> PROFIT
  Gamma earnings < Theta cost  when  RV < IV  --> LOSS

Short straddle (negative Gamma):
  Exact opposite. Profit when RV < IV, lose when RV > IV.
```

This is why the paper calls it "Gamma-Theta relationship" -- Gamma is the profit mechanism and Theta is the cost, and RV vs IV determines which side wins.

---

## Part 10: PnL Decomposition (Understanding Trade Results)

The paper shows trade PnL broken into components:

```
Total PnL = Delta PnL + Gamma PnL + Theta PnL + Vega PnL

Example: Long straddle held for 24 hours

  Gamma PnL:  +0.03 BTC  (BTC moved a lot, gamma scalping profitable)
  Vega PnL:   +0.01 BTC  (IV increased during holding period)
  Theta PnL:  -0.015 BTC (24 hours of time decay)
  Delta PnL:  -0.005 BTC (small residual from imperfect hedging)
  ------------------------------------------------
  Total PnL:  +0.02 BTC  (net profit)
```

For a typical SHORT straddle:
```
  Theta PnL:  +0.015 BTC (earned time decay -- the "premium")
  Vega PnL:   +0.005 BTC (IV decreased -- fear subsided)
  Gamma PnL:  -0.008 BTC (small price moves caused gamma losses)
  Delta PnL:  -0.002 BTC (hedging friction)
  ------------------------------------------------
  Total PnL:  +0.01 BTC  (net profit from calm market)
```

---

## Part 11: Evaluation Metrics (How Success is Measured)

### Profit Metrics

**Total Return (TR):**
```
TR = (Final Value - Initial Value) / Initial Value

Started with 10 BTC, ended with 13.31 BTC
TR = (13.31 - 10) / 10 = 33.1%
```

### Risk-Adjusted Metrics

**Sharpe Ratio (ASR):**
```
ASR = (Average daily return - Risk-free rate) / Std(daily returns) * sqrt(365)

"How much return do I get per unit of risk?"
  ASR > 1.0: Good
  ASR > 1.5: Very good
  ASR > 2.0: Excellent
  OPHR BTC: 1.87 (very good)
```

**Calmar Ratio (ACR):**
```
ACR = Annualized return / Maximum Drawdown

"How much return do I get relative to my worst loss?"
  OPHR BTC: 3.35 (excellent -- earns 3.35x its worst drawdown per year)
```

**Sortino Ratio (ASoR):**
```
Like Sharpe but only penalizes DOWNSIDE volatility (not upside)
"Upside volatility is GOOD, only punish me for losses"
  OPHR BTC: 3.27 (excellent)
```

### Risk Metrics

**Annual Volatility (AVOL):**
```
How much daily returns fluctuate, annualized
  OPHR BTC: 16.83% (moderate -- not too wild)
```

**Maximum Drawdown (MDD):**
```
Worst peak-to-trough loss

If your portfolio went: $10 -> $12 -> $10.88 -> $13
  Peak was $12, trough was $10.88
  MDD = ($12 - $10.88) / $12 = 9.33%
  
OPHR BTC: 9.41% (low -- never lost more than ~9.4% from a peak)
```

### Trade Metrics

**Win Rate (WR):**
```
% of individual trades that were profitable
  OPHR BTC: 45.93%

Note: Win rate below 50% can still be very profitable if winning trades
are much larger than losing trades (which is the case -- PLR = 2.05)
```

**Profit/Loss Ratio (PLR):**
```
Average winning trade size / Average losing trade size
  OPHR BTC: 2.05

"When I win, I win $2.05 for every $1 I lose when I'm wrong"
Combined with 45.93% WR: Expected value per trade is positive
```

**Holding Period (HP):**
```
Average time a position is held
  OPHR Long trades:  21 hours (quick volatility captures)
  OPHR Short trades: 51 hours (patient premium collection)
```

---

## Part 12: Common Viva Questions on Finance

**Q: Why use straddles and not just calls or puts?**
> Calls profit from price going UP. Puts profit from price going DOWN. We don't want to bet on direction -- we want to bet on VOLATILITY (how much the price moves regardless of direction). A straddle combines both and is delta-neutral, isolating the volatility bet.

**Q: Why crypto options specifically?**
> Crypto markets are transparent (all Deribit data is accessible), operate 24/7 (no market close gaps), have high volatility (more trading opportunities), and have well-defined fee structures. Traditional equity options data is harder to obtain and has more complex market microstructure.

**Q: What happens if the agent makes a bad trade?**
> The margin system protects against catastrophic losses. If a trade would exceed the portfolio's margin capacity, the environment rejects it. The maximum loss is bounded by the initial capital (10 BTC). During backtesting, OPHR's worst drawdown was only 9.41% on BTC.

**Q: Why does "always buy options" lose -33%?**
> Time decay (theta). Buying options means paying theta every hour. Unless the market moves enough (RV > IV), the theta cost exceeds the gamma profit. Most of the time, IV overestimates RV, so buyers lose on average.

**Q: Why does "always sell options" only make +2.9% despite winning 73.9% of trades?**
> Tail risk. When selling options, you win small amounts frequently (theta income) but occasionally get destroyed by large moves. One catastrophic loss can erase months of small gains. This is the classic "picking up pennies in front of a steamroller" problem.

**Q: What is "gamma scalping"?**
> When you're long gamma and the market oscillates, each delta hedge is effectively "buy low, sell high." Price drops -> delta turns negative -> you buy BTC. Price rises -> delta turns positive -> you sell BTC. Each round trip locks in a small profit. This is the hedging profit mechanism for long gamma positions.

# Presentation Script: OPHR - Mastering Volatility Trading with Multi-Agent Deep RL

**Format:** 10 minutes presentation + 5 minutes Q&A  
**Slide count:** 12-14 slides recommended

---

## SLIDE 1: Title Slide (30 seconds)

**Title:** OPHR: Mastering Volatility Trading with Multi-Agent Deep Reinforcement Learning  
**Subtitle:** NeurIPS 2025 | Chen, Cai, Qin, An (NTU / Skywork AI)

**Script:**
> "Good [morning/afternoon]. Today we are presenting our analysis of OPHR, a paper published at NeurIPS 2025. This paper introduces the first reinforcement learning framework for volatility trading through options -- a problem that combines two of the hardest challenges in quantitative finance: predicting volatility regimes and managing the risk of option positions through dynamic hedging."

---

## SLIDE 2: What is Volatility Trading? (1 minute)

**Content:**
- Options prices embed Implied Volatility (IV) -- the market's expectation of future price swings
- Realized Volatility (RV) -- the actual price fluctuations that occur
- When RV > IV: buying options (long gamma) is profitable
- When RV < IV: selling options (short gamma) is profitable
- Key equation: PnL = integral of [Gamma * (RV^2 - IV^2)] dt
- Diagram: IV vs RV timeline with long/short zones marked

**Script:**
> "To understand OPHR, we first need to understand volatility trading. Options prices contain an implied volatility -- what the market thinks future price swings will be. Realized volatility is what actually happens. When RV exceeds IV, buying options is profitable; when IV exceeds RV, selling options earns a premium. This is captured mathematically by this equation, which shows that the PnL of a delta-hedged option position is the integral of gamma times the difference between realized and implied variance. The challenge is twofold: first, you need to predict which regime you are in, and second, you need to hedge your positions correctly to realize that profit."

---

## SLIDE 3: Why is This Hard? (1 minute)

**Content:**
- Path dependency: Same RV, different PnL depending on price path (show Figure 1 from paper)
  - Oscillating market: hedge at extremes, lock in profits
  - Trending market: hedging limits gains
- Black-Scholes assumptions violated in practice: no constant volatility, transaction costs exist, cannot hedge continuously
- Crypto options: 24/7 markets, extreme tails, regime shifts

**Script:**
> "This figure from the paper perfectly illustrates why hedging strategy matters. Both scenarios have the same realized volatility, but the PnL outcomes are completely different. On the left, the market oscillates -- price goes down then up then back -- and hedging at each extreme locks in profit. On the right, the market trends consistently upward, and hedging actually limits your gains. So selecting the RIGHT hedging strategy at the RIGHT time is just as important as the initial trading decision. Traditional models like Black-Scholes assume constant volatility and continuous hedging, which clearly don't hold in practice, especially in crypto markets."

---

## SLIDE 4: OPHR Architecture Overview (1.5 minutes)

**Content:**
- Diagram showing the two-agent architecture (reproduce Figure 2 from paper)
- OP-Agent: Observes market features -> Decides Long/Short/Neutral
- HR-Agent: Observes market + position + Greeks -> Selects hedger from pool
- Hedger Pool: Multiple hedging strategies with different risk preferences
- Cooperative MDP: Both agents share the objective of maximizing portfolio net value

**Script:**
> "OPHR solves this with a multi-agent architecture. There are two specialized agents. The OP-Agent, or Option Position Agent, observes market features like the volatility surface, price momentum, and volume, and decides whether to go long volatility, short volatility, or stay neutral. It outputs one of three actions every hour. The HR-Agent, the Hedger Routing Agent, takes a richer state including the current position and all four Greeks -- delta, gamma, theta, and vega -- and selects which hedging strategy to use from a pool of 8 hedgers with different risk levels. It makes this decision every 24 hours. These agents cooperate: the OP-Agent's position decisions determine what the HR-Agent needs to hedge, and the HR-Agent's hedging quality affects the OP-Agent's returns."

---

## SLIDE 5: Training -- Phase 1: Oracle Initialization (1 minute)

**Content:**
- Oracle Policy: Uses future RV (cheating!) to generate trading signals
  - If future RV >= IV * (1 + 0.1): go long
  - If future RV <= IV * (1 - 0.1): go short
  - Otherwise: neutral
- Why sub-optimal: fixed threshold, ignores path dependency, uses baseline hedger
- Purpose: Bootstrap OP-Agent's replay buffer with high-quality experience
- Solves cold-start problem -- options rewards are delayed and noisy

**Script:**
> "Training happens in two phases. Phase 1 uses an Oracle policy that has access to future realized volatility -- something impossible in production. The Oracle compares future RV to current IV and generates long/short/neutral signals. It's deliberately sub-optimal because it uses a fixed threshold and ignores path-dependent effects, but it provides a good initialization. We collect experience from the Oracle and fill the OP-Agent's replay buffer. This is crucial because without this warm start, the agent would need to explore a massive action space with very sparse, delayed rewards -- option positions might take 12 to 50 hours to show their true value."

---

## SLIDE 6: Training -- Phase 2: Iterative Training (1 minute)

**Content:**
- Alternating training: OP-Agent trains -> HR-Agent trains -> repeat (5 iterations)
- OP-Agent: n-step Double DQN (n=12 hours), epsilon-greedy exploration
- HR-Agent: Twin Environment technique
  - Save state -> Run selected hedger -> Get value V
  - Restore state -> Run baseline hedger -> Get value V_hat
  - Reward = V - V_hat (relative improvement over baseline)
- Why iterate: Better hedging enables more aggressive positions, which require better hedging

**Script:**
> "Phase 2 is where the real learning happens. We alternate between training the OP-Agent and the HR-Agent for 5 iterations. The OP-Agent uses n-step Double DQN with n=12, meaning it looks 12 hours ahead to evaluate each decision. The HR-Agent uses a clever twin environment technique: at each decision point, we save the environment state, run forward 24 hours with the selected hedger, note the portfolio value, then rewind and run the same 24 hours with the baseline hedger. The HR-Agent's reward is the difference -- how much better was the selected hedger than the baseline? This iterative training is essential because the two agents co-adapt: a better hedger lets the OP-Agent take more aggressive positions, and those aggressive positions require the HR-Agent to learn even more sophisticated hedging."

---

## SLIDE 7: Key Mathematical Details (1 minute)

**Content:**
- n-step TD target: Q_target = sum(gamma^k * r_k, k=0..n-1) + gamma^n * Q_target(s_n, argmax_a Q(s_n, a))
- Double DQN: Online network selects action, target network evaluates
- State representation:
  - OP: 96 dims (48 vol surface + 48 perpetual features)
  - HR: 102 dims (96 market + 2 position + 4 Greeks)
- Action spaces: OP = {long, neutral, short}, HR = {hedger_0 ... hedger_7}

**Script:**
> "The OP-Agent uses this n-step TD learning objective. Instead of bootstrapping from the next state, it accumulates 12 hours of discounted rewards before bootstrapping. This is critical for options where a single timestep reward is dominated by noise from theta decay and hedging costs. The Double DQN formulation prevents overestimation: the online network picks the best action, but the target network evaluates it. The state spaces are 96 dimensions for the OP-Agent, combining volatility surface features with perpetual futures features, and 102 dimensions for the HR-Agent, adding position and Greeks information."

---

## SLIDE 8: Results (1 minute)

**Content:**
- Table 2 from paper (key rows only): OPHR vs top baselines
  - BTC: OPHR 33.10% TR, 1.87 ASR, 9.41% MDD, 3.35 ACR
  - ETH: OPHR 44.89% TR, 1.76 ASR, 27.25% MDD, 1.58 ACR
  - Best baseline (DLOT): 4.91% BTC, 1.19% ETH
  - All ML baselines negative returns
- OP-only vs OPHR: HR-Agent adds ~12% return on BTC
- Transaction costs: 9.36% (BTC), 5.75% (ETH) of PnL -- strategy is robust to costs

**Script:**
> "The results speak for themselves. OPHR achieves 33% return on BTC and 45% on ETH with strong risk-adjusted metrics. Every single baseline has negative or near-zero returns except DLOT. The OP-Agent alone gets 21% on BTC, and adding the HR-Agent pushes this to 33%, showing the hedger routing adds real value. Importantly, transaction costs consume less than 10% of PnL, confirming the strategy is not dependent on unrealistic cost assumptions."

---

## SLIDE 9: Closer Look -- Trading Behavior (45 seconds)

**Content:**
- Table 3: Long vs Short holding periods
  - Long positions: ~9-21 hours (quick, during volatility spikes)
  - Short positions: ~50 hours (patient, during calm periods)
- HR-Agent improves Win Rate by 1.7-2.9% and PLR by 0.07-0.22 across both position types
- Figure 3: Trade examples showing PnL decomposition (Gamma/Theta/Delta/Vega)

**Script:**
> "Looking deeper at the trading behavior, OPHR shows sophisticated volatility timing. Long positions are held for about 9 to 21 hours -- it quickly enters when volatility spikes and exits before theta decay erodes profits. Short positions are held for about 50 hours, patiently collecting theta premium during calm markets. The HR-Agent consistently improves both Win Rate and Profit/Loss Ratio across position types, confirming it adapts its hedging to the market regime."

---

## SLIDE 10: Live Demo (1 minute)

**Content:** Switch to Jupyter Notebook / Terminal

**Script:**
> "Let me now show you the code in action. [Switch to screen]
> 
> Here we have the repository structure. Let me walk you through the key components:
> 
> [Show agents/op_agent.py] This is the OP-Agent -- a standard DQN with the n-step buffer. The Q-Network takes 96-dimensional input and outputs 3 Q-values for long, neutral, and short.
> 
> [Show training/phase2_iterative.py] This is the iterative training loop. You can see the twin environment mechanism here -- save_state, run selected hedger, restore_state, run baseline, compute relative reward.
> 
> [Show backtest results / plots if available] Here are the backtest results showing the PnL curve, delta exposure (which stays near zero thanks to hedging), and the position timeline."

---

## SLIDE 11: Critical Evaluation (1 minute)

**Content:**
- **Strengths:**
  - First RL framework for volatility trading -- novel problem formulation
  - Realistic environment with actual exchange margin model and fees
  - Principled multi-agent decomposition mirrors real trading desk structure
  - Oracle initialization elegantly solves the cold-start problem
- **Limitations:**
  - Single train/test split -- no error bars or cross-validation
  - Crypto-only evaluation -- unclear if generalizes to equity/FX options
  - Fixed ATM straddle -- professional traders use diverse strategies
  - No online adaptation for non-stationary markets
  - Oracle may introduce subtle look-ahead bias

**Script:**
> "For our critical evaluation: the paper's key strength is the principled problem decomposition. Separating volatility timing from hedging mirrors how real trading desks operate and makes the learning problem tractable. The realistic environment with actual Deribit margin calculations is unusual for academic papers and strengthens the results. However, there are limitations. The paper uses a single train-test split with no error bars, making it hard to assess robustness. It only tests on crypto -- the dynamics of equity options are very different. The framework is limited to ATM straddles when professional traders use many more structures. And there is no mechanism for online adaptation when market regimes shift."

---

## SLIDE 12: Real-World Application Proposal (30 seconds)

**Content:**
- Proposed: Catastrophe Bond Portfolio Management
  - Similar volatility premium structure (earn spread in calm, risk catastrophe)
  - Multi-agent decomposition maps naturally (exposure vs. hedging)
  - Modifications needed: climate features, catastrophe loss models, CVaR rewards

**Script:**
> "For a novel application, we propose adapting OPHR for catastrophe bond portfolio management. Cat bonds have a similar structure to short volatility -- investors earn a premium during calm periods but face extreme losses during natural disasters. The OP-Agent could time exposure to different perils, while the HR-Agent could route between hedging instruments like weather derivatives and reinsurance contracts. The key modification would be replacing market features with climate indicators and using CVaR-based objectives for the tail-heavy loss distribution."

---

## SLIDE 13: Conclusion (20 seconds)

**Content:**
- OPHR is a significant contribution: first RL framework for volatility trading
- Multi-agent architecture with Oracle initialization is the key innovation
- Results are strong but need more robustness testing
- Opens the door for RL in complex derivatives trading

**Script:**
> "In conclusion, OPHR represents a significant advancement in applying reinforcement learning to financial markets. Its multi-agent architecture with Oracle-guided initialization provides an elegant solution to the complex problem of volatility trading. While the results need more robustness testing, this paper opens an exciting new direction for RL in derivatives markets. Thank you -- we are happy to take questions."

---

## Q&A PREPARATION: Likely Questions and Answers

### Q1: "Why not use a single agent for both decisions?"
> "The position and hedging decisions operate on fundamentally different time scales -- OP decides every hour, HR every 24 hours -- and require different state information. A single agent would face a much larger joint action space (3 positions x 8 hedgers = 24 actions) with an augmented state, making learning harder. The decomposition also enables modular improvements -- you can improve the hedger pool without retraining the OP-Agent."

### Q2: "Isn't using future RV in the Oracle cheating?"
> "Yes, the Oracle is explicitly labeled as sub-optimal and only used for initialization. It is like a teacher who knows the answers but doesn't teach perfectly. The key insight is that the Oracle's experience fills the replay buffer with transitions that represent profitable behavior patterns. The RL agent then learns to replicate and improve upon these patterns without access to future information. In Phase 2, the agent trains online with only current market data."

### Q3: "How do you know this isn't overfitting to the test period?"
> "This is a valid concern. The paper uses data from 2019-2022 for training, 2023 H1 for validation, and 2023 H2 to 2024 H1 for testing. These periods span different market regimes including bull markets, bear markets, and high-volatility events. However, we acknowledge that a single split without error bars is a limitation. Ideally, we'd want walk-forward analysis or multiple random seeds."

### Q4: "What happens when the market regime changes significantly?"
> "The current framework deploys fixed weights after training. In a regime shift (e.g., from crypto bear market to bull), performance could degrade. A practical enhancement would be periodic retraining with a sliding window, or implementing online fine-tuning where the agents continue learning from recent experience while trading."

### Q5: "Can you explain the n-step TD update in simple terms?"
> "Instead of asking 'how good is my action based on the next step's reward,' n-step TD asks 'how good is my action based on the next 12 hours of actual rewards?' For options, this is essential because a decision to sell a straddle might look bad after 1 hour (small negative theta), decent after 6 hours, and clearly profitable after 12 hours when the full theta has accrued. The 12-step window captures this delayed payoff."

### Q6: "What is the computational cost?"
> "Moderate. The neural networks are simple 2-layer MLPs with 1024 hidden units -- no GPUs required. The main cost is environment simulation, since each step requires margin calculations across 35 scenarios. On a standard laptop CPU, you can train a reduced version in a few hours. The full paper setup with 1000+ episodes would take longer but is feasible on Google Colab."

### Q7: "How does the hedger pool work?"
> "The pool contains 8 hedgers: 5 delta-threshold hedgers with thresholds from 0.05 (very aggressive, hedges tiny delta deviations) to 3.0 (effectively never hedges), and 3 price-based hedgers that trigger based on price movement percentage. The HR-Agent learns which hedger is best for the current market conditions. In volatile markets, it tends to select aggressive hedgers; in calm markets, it selects conservative ones to minimize transaction costs."

### Q8: "What are the key differences between OPHR and Deep Hedging?"
> "Deep Hedging focuses only on the hedging problem -- given an existing option position, how to hedge it optimally. OPHR adds the position management layer on top: when to enter/exit positions and how to dynamically route between different hedging strategies. Deep Hedging trains on simulated data; OPHR trains on real market data. Deep Hedging is a single-agent problem; OPHR is multi-agent."

### Q9: "What is the reward function?"
> "The OP-Agent's reward is the change in portfolio net value: r = V(t+1) - V(t). This is the raw dollar PnL at each step. The HR-Agent's reward is relative: r = V_selected - V_baseline, measuring how much better the selected hedger performed than the baseline over a 24-hour window. The relative reward is crucial because it isolates the hedging contribution from the overall position PnL."

### Q10: "If asked to modify the code during viva -- what could you change?"
> "Common modifications: (1) Change the n-step parameter in training_config.yaml from 12 to 6 or 24. (2) Add a new hedger to the pool in hr_agent.py. (3) Modify the Oracle beta threshold in oracle_policy.py. (4) Change the epsilon decay schedule. (5) Adjust the reward function in rl_env.py to include a risk penalty."

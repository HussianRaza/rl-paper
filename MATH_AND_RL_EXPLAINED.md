# OPHR -- Math & Reinforcement Learning Explained Simply

---

## Part 1: Reinforcement Learning Basics (What You Need to Know)

### What is RL?

RL is about an **agent** learning to make good **decisions** by trial and error.

```
Agent sees a SITUATION (state)
  --> picks an ACTION
  --> gets a REWARD (good or bad)
  --> sees the NEXT SITUATION
  --> repeats

Goal: Learn which actions give the most total reward over time
```

**Real-life analogy:** A child learning to ride a bicycle.
- State: balance, speed, angle of handlebars
- Action: turn left, turn right, pedal, brake
- Reward: +1 for staying upright, -100 for falling
- Over many attempts, the child learns the best actions for each situation

### The MDP (Markov Decision Process)

An MDP is the mathematical framework for RL. It has 5 parts:

```
MDP = (S, A, T, R, gamma)

S = States      --> All possible situations the agent can be in
A = Actions     --> All possible things the agent can do
T = Transitions --> How the world changes after an action (S x A -> S')
R = Reward      --> How good/bad each action was (S x A -> number)
gamma = Discount --> How much the agent cares about future vs now (0 to 1)
```

**In OPHR:**
```
OP-Agent MDP:
  S = Market features (96 numbers: volatility surface + price data)
  A = {Long, Neutral, Short}  (3 choices)
  T = Market moves to next hour (agent can't control this)
  R = Change in portfolio value (V_t+1 - V_t)
  gamma = 0.99 (cares a lot about future rewards)

HR-Agent MDP:
  S = Market features + position + Greeks (102 numbers)
  A = {Hedger_0, Hedger_1, ..., Hedger_7}  (8 choices)
  T = Market moves + position changes over 24 hours
  R = V_selected_hedger - V_baseline_hedger (relative improvement)
  gamma = 0.99
```

### Cooperative MDP (Two Agents Working Together)

OPHR uses a **cooperative** MDP -- two agents share the same goal (maximize portfolio value).

```
Formally:
  (MDP_op, MDP_hr) where both optimize the same portfolio

Why cooperative, not independent:
  - OP-Agent's positions CHANGE what the HR-Agent needs to hedge
  - HR-Agent's hedging quality AFFECTS the OP-Agent's returns
  - They MUST coordinate for good results
```

Think of it like two players in a co-op video game. The attacker (OP) decides what enemies to engage, and the healer (HR) decides how to keep the team alive. They share the same score.

---

## Part 2: Q-Learning and DQN

### Q-Values (The Core Idea)

A **Q-value** Q(s, a) answers: "If I'm in state s and take action a, then act optimally forever after, what's my expected total future reward?"

```
Example:
  State: Market is volatile, IV is low
  Q(state, Long)    = 0.85  --> "Going long here is expected to earn 0.85"
  Q(state, Neutral) = 0.20  --> "Doing nothing is expected to earn 0.20"
  Q(state, Short)   = -0.30 --> "Going short is expected to lose 0.30"

  Best action: Long (highest Q-value)
```

### The Bellman Equation (How Q-Values Update)

The fundamental equation of Q-learning:

```
Q(s, a) = r + gamma * max_a' Q(s', a')

In words:
  The value of taking action a in state s
  = immediate reward r
  + discounted value of the best action in the next state

Example:
  Current state: calm market
  Action: sell options (short)
  Immediate reward: +0.01 (small theta income)
  Next state: still calm
  Best next action value: +0.80

  Q(calm, short) = 0.01 + 0.99 * 0.80 = 0.802
```

This is computed iteratively -- each time the agent experiences a transition, it updates its Q-value estimate to be more accurate.

### DQN (Deep Q-Network)

When the state space is huge (96 or 102 continuous numbers), you can't store a Q-value for every possible state. Instead, use a **neural network** to approximate Q-values:

```
Neural Network:
  Input:  state (96 numbers) 
  Output: Q-values for each action (3 numbers)

  [market features] --> [1024 neurons] --> [1024 neurons] --> [Q(long), Q(neutral), Q(short)]
                        ReLU activation   ReLU activation

Architecture in code (op_agent.py):
  Linear(96, 1024) -> ReLU -> Linear(1024, 1024) -> ReLU -> Linear(1024, 3)
```

**ReLU activation:** A simple function: output = max(0, input). It allows the network to learn non-linear patterns. If input is negative, output is 0. If positive, output equals input.

**Training:** The network learns by minimizing the difference between its predicted Q-value and the "target" Q-value (from the Bellman equation):

```
Loss = (Q_predicted - Q_target)^2

Q_predicted = Q_network(state)[action]  
Q_target    = reward + gamma * max Q_target_network(next_state)

The network adjusts its weights to make Q_predicted closer to Q_target
```

### Experience Replay Buffer

The agent doesn't learn from one experience at a time. It stores thousands of past experiences and randomly samples batches for training:

```
Buffer = [(state_1, action_1, reward_1, next_state_1, done_1),
          (state_2, action_2, reward_2, next_state_2, done_2),
          ...
          (state_100000, ...)]

Each training step:
  Randomly sample 512 experiences from the buffer
  Compute loss on all 512 at once
  Update neural network weights
```

**Why random sampling?** Consecutive experiences are correlated (hour 5 is very similar to hour 6). Training on correlated data makes neural networks learn poorly. Random sampling breaks this correlation.

### Double DQN (Used in OPHR)

Regular DQN has a problem: it **overestimates** Q-values. The "max" operation picks the highest Q-value, which is often the one with the most positive noise, not the truly best action.

**Fix -- use TWO networks:**

```
Regular DQN:
  Q_target = reward + gamma * max Q_target_net(next_state)
                                 ^^^
                    Target network picks action AND evaluates it
                    (same network = biased upward)

Double DQN:
  best_action = argmax Q_online_net(next_state)     <-- Online picks action
  Q_target = reward + gamma * Q_target_net(next_state, best_action)  <-- Target evaluates it
                                                     
  Two different networks = less bias
```

**In the code (op_agent.py, line 149-151):**
```python
next_actions = self.q_network(next_state_batch).argmax(1)          # Online net picks
next_q = self.target_network(next_state_batch).gather(1, next_actions)  # Target net evaluates
```

### Target Network

The target network is a **slowly-updated copy** of the main network:

```
Every 10 training steps:
  target_network.weights = main_network.weights  (hard copy)

Why: If the target changes every step, you're chasing a moving goalpost.
     A stable target makes learning smoother.
```

---

## Part 3: n-step TD Learning (Critical for OPHR)

### The Problem with 1-step

In standard (1-step) TD learning:
```
Q_target = r_1 + gamma * V(s_1)

"How good was my action? Look at the IMMEDIATE reward + estimate of future."
```

For options trading, this is terrible because:
- At hour 0, you sell a straddle
- Hour 1 reward: -0.001 (tiny theta gain, overshadowed by noise)
- Hour 6 reward: +0.003 (theta accumulating)
- Hour 12 reward: +0.015 (a full day of theta earned)

The 1-step reward at hour 1 tells you almost nothing about whether selling was a good decision.

### The n-step Solution (n=12 in OPHR)

Instead of looking 1 step ahead, look **12 steps ahead**:

```
1-step:   Q_target = r_1 + gamma * V(s_1)
12-step:  Q_target = r_1 + gamma*r_2 + gamma^2*r_3 + ... + gamma^11*r_12 + gamma^12 * V(s_12)

In words: "Add up the actual rewards for 12 hours, THEN estimate the future."
```

**Why 12?** Options positions in OPHR are typically held 9-50 hours. A 12-hour window captures a significant portion of the position's lifecycle, giving a much clearer signal about whether the decision was good.

**Concrete example:**
```
Hour 0: Sell straddle (short gamma)
Hour 1:  r = +0.001 (theta)
Hour 2:  r = +0.001 
Hour 3:  r = -0.005 (small price spike, gamma loss)
Hour 4:  r = +0.002 (price returns)
...
Hour 12: r = +0.001

12-step return = 0.001 + 0.99*0.001 + 0.99^2*(-0.005) + ... + 0.99^12 * V(s_12)
             = +0.008 (positive! Selling was a good decision over 12 hours)

vs 1-step return = 0.001 + 0.99 * V(s_1) 
                (unreliable -- too noisy to learn from)
```

**The n-step buffer in code (replay_buffer.py):**
```python
class NStepBuffer:
    def __init__(self, n_step=12, gamma=0.99):
        self.buffer = deque(maxlen=12)   # Sliding window of 12 transitions
    
    def get_n_step_transition(self):
        # Compute: R = r_0 + 0.99*r_1 + 0.99^2*r_2 + ... + 0.99^11*r_11
        n_step_reward = 0
        for i, (_, _, r, _, done) in enumerate(self.buffer):
            n_step_reward += (0.99 ** i) * r   # Discounted sum
        return (state_0, action_0, n_step_reward, state_12, done_12)
```

### The Full n-step TD Loss (Equation 5 from the Paper)

```
L(theta) = E[ (  sum_{k=0}^{11} gamma^k * r_{j+k}                    <-- 12 actual rewards
              +  gamma^12 * Q_target(s_{j+12}, argmax_a Q(s_{j+12}, a)) <-- Bootstrap after 12 steps (Double DQN)
              -  Q(s_j, a_j)                                            <-- Current estimate
              )^2 ]                                                     <-- Squared error

Step by step:
1. Take action a_j in state s_j
2. Collect 12 rewards: r_j, r_{j+1}, ..., r_{j+11}
3. Arrive at state s_{j+12}
4. Compute discounted sum of 12 rewards
5. Add the discounted estimated future value at s_{j+12} (using Double DQN)
6. Subtract current Q-value estimate
7. Square the difference = loss
8. Minimize loss by adjusting network weights
```

---

## Part 4: Epsilon-Greedy Exploration

The agent needs to **explore** (try random actions to discover new strategies) vs **exploit** (use what it already knows works).

```
Epsilon-greedy strategy:
  With probability epsilon: take a RANDOM action (explore)
  With probability 1-epsilon: take the BEST action according to Q-values (exploit)

Epsilon schedule in OPHR:
  Start: epsilon = 0.9  (90% random -- lots of exploration early)
  End:   epsilon = 0.01 (1% random -- mostly exploitation later)
  Decay: epsilon = epsilon * 0.995 after each episode

Timeline:
  Episode 1:    epsilon = 0.90 (explore heavily)
  Episode 100:  epsilon = 0.55 (balanced)
  Episode 500:  epsilon = 0.08 (mostly exploit)
  Episode 1000: epsilon = 0.01 (almost pure exploitation)
```

During backtesting/evaluation, epsilon = 0 (always pick the best action, no randomness).

---

## Part 5: The Discount Factor (gamma = 0.99)

Gamma controls how much the agent cares about future rewards vs immediate rewards:

```
gamma = 0:    Only cares about immediate reward (extremely short-sighted)
gamma = 0.5:  Reward 10 steps away is worth 0.5^10 = 0.001 of its face value
gamma = 0.99: Reward 10 steps away is worth 0.99^10 = 0.904 of its face value
gamma = 1:    All future rewards are equally important (can be unstable)
```

OPHR uses gamma = 0.99 because options positions play out over many hours. A reward 24 hours from now is worth 0.99^24 = 0.786 of its face value -- still highly relevant.

---

## Part 6: The HR-Agent's Reward (Relative Reward)

The HR-Agent doesn't use the raw PnL as reward. It uses a **relative** reward:

```
r_HR = V_selected - V_baseline

Where:
  V_selected = portfolio value after 24h using the hedger chosen by HR-Agent
  V_baseline = portfolio value after 24h using the default hedger (delta threshold = 0.1)
```

**Why relative?**
```
Imagine two scenarios:

Scenario A: Market crashes 20%
  V_selected = $8,500 (selected hedger lost $1,500)
  V_baseline = $8,000 (baseline lost $2,000)
  r_HR = +$500  (selected hedger was BETTER, even though both lost money)

Scenario B: Market is calm
  V_selected = $10,050 (selected hedger gained $50)
  V_baseline = $10,030 (baseline gained $30)
  r_HR = +$20  (selected hedger was slightly better)

If we used raw PnL:
  Scenario A reward = -$1,500 (terrible!)
  Scenario B reward = +$50 (good!)

But the HR-Agent made a BETTER decision in Scenario A than B!
The relative reward correctly captures this.
```

The relative reward removes the effect of the market direction and the option position, isolating ONLY the hedging contribution.

---

## Part 7: Putting It All Together -- Training Flow

```
PHASE 1: Oracle Initialization
==================================
Oracle (has future info) generates trades
  |
  v
Store (state, action, reward, next_state) in OP-Agent's replay buffer
  |
  v
Train OP-Agent offline on this data using 12-step Double DQN
  |
  v
Now OP-Agent has a "warm start" -- it knows roughly what good trading looks like


PHASE 2: Iterative Training (repeat 5 times)
==============================================

  Iteration i:
  
  Step A: Train OP-Agent (200 episodes)
  -----------------------------------------
  for each episode:
    for each hour:
      1. OP-Agent sees state (96 features)
      2. Epsilon-greedy: random action with prob epsilon, else argmax Q(s,a)
      3. HR-Agent (FROZEN) selects hedger based on its current policy
      4. Environment executes option trade + hedge
      5. Get reward = V(t+1) - V(t)
      6. Store in n-step buffer -> after 12 steps -> store in replay buffer
      7. Every 1 step: sample batch from buffer, compute loss, update Q-network
      8. Every 10 steps: copy Q-network weights to target network
      9. Decay epsilon slightly
  
  Step B: Train HR-Agent (50 episodes)
  -----------------------------------------
  for each episode:
    every 24 hours:
      1. HR-Agent sees state (102 features)
      2. Epsilon-greedy: pick random hedger or best hedger
      3. --- TWIN ENVIRONMENT ---
         a. Save environment state
         b. Run 24h with selected hedger --> V_selected
         c. Restore state
         d. Run 24h with baseline hedger --> V_baseline
         e. r_HR = V_selected - V_baseline
         f. Restore state, continue with selected hedger
      4. Store (state, hedger_idx, r_HR, next_state) in buffer
      5. Sample batch, compute DQN loss, update HR Q-network
      6. Decay epsilon

Final: Save both trained agents as checkpoint files
```

---

## Part 8: Key Equations Summary Card

```
1. BELLMAN EQUATION (foundation of all Q-learning):
   Q(s,a) = r + gamma * max_a' Q(s', a')

2. DQN LOSS:
   Loss = (Q_network(s,a) - target)^2
   where target = r + gamma * max Q_target_net(s')

3. DOUBLE DQN LOSS:
   Loss = (Q_network(s,a) - target)^2
   where target = r + gamma * Q_target_net(s', argmax_a Q_network(s', a))
                                                 ^^ online picks    ^^ target evaluates

4. N-STEP TD TARGET (n=12):
   target = [r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^11*r_11]
            + gamma^12 * Q_target_net(s_12, argmax_a Q_network(s_12, a))

5. EPSILON-GREEDY:
   action = random          with probability epsilon
   action = argmax Q(s,a)   with probability 1-epsilon

6. HR-AGENT REWARD:
   r_HR = V(t+24, selected_hedger) - V(t+24, baseline_hedger)

7. OP-AGENT REWARD:
   r_OP = V(t+1) - V(t)    (net value change)

8. EPSILON DECAY:
   epsilon_new = max(0.01, epsilon_old * 0.995)
```

---

## Part 9: Common Viva Questions on Math/RL

**Q: What is the difference between TD(0) and n-step TD?**
> TD(0) uses 1 actual reward + bootstraps from next state. n-step TD uses n actual rewards + bootstraps from the state n steps later. n-step has lower bias (more real rewards) but higher variance (more randomness in n rewards). n=12 is chosen because options profits unfold over 12+ hours.

**Q: Why DQN and not Policy Gradient or Actor-Critic?**
> The action spaces are small and discrete (3 actions for OP, 8 for HR). DQN works well for discrete actions. Policy gradient methods are better for continuous actions. Also, DQN with experience replay is more sample-efficient, which matters when each episode requires expensive environment simulation.

**Q: What does the target network prevent?**
> Without it, the Q-value estimate and the target it's trying to match both change simultaneously, causing unstable oscillations (chasing a moving target). The target network provides a stable goal by updating less frequently.

**Q: Why gamma = 0.99 and not 0.9 or 1.0?**
> 0.99 means a reward 100 steps away is worth 0.99^100 = 0.366 of its value. Options positions last 9-50 hours, so we need to care about rewards far into the future. gamma=0.9 would discount too aggressively (0.9^24 = 0.08, almost ignoring rewards a day later). gamma=1.0 can cause infinite returns in non-terminating episodes.

**Q: How does the cooperative MDP differ from independent learners?**
> Independent learners each have their own reward and ignore each other. In OPHR's cooperative MDP, both agents share the same ultimate goal (portfolio value), and each agent's transition function depends on the other's actions. The OP-Agent's position changes what the HR-Agent sees, and the HR-Agent's hedging affects the OP-Agent's rewards.

**Q: What is the loss function being minimized?**
> Mean Squared Error between the current Q-value estimate and the target: L = E[(Q(s,a) - target)^2]. The target is the discounted sum of n rewards plus the bootstrapped value. Adam optimizer with learning rate 0.0001 adjusts the network weights to minimize this loss.

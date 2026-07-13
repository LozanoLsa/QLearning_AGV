# Project 22 · AGV Route Intelligence
### Tabular Q-Learning for Autonomous Navigation on a Factory Floor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/QLearning_AGV/blob/main/22_QLearning_AGV.ipynb)

> *"The agent doesn't learn from data. It learns from consequence — and that changes everything."*

---

For twenty-one projects, every model in this portfolio was handed a dataset and asked to find a pattern. The intelligence was retrospective: trained on what had already happened, applied to what happens next.

This project removes the dataset entirely.

What remains is a **world**, a **reward signal**, and a mandate to figure the rest out alone.

---

## Episode 0 — The World Before Training

The environment is a **11×11 factory floor**: 121 cells, 4 possible moves, one destination.

```
┌────────────────────────────────────────────┐
│  Grid      : 11 × 11  =  121 states        │
│  Actions   : Up · Down · Left · Right       │
│  Obstacles : 11 fixed walls and equipment   │
│  Congestion: 3 high-traffic aisles          │
│  Goal      : Delivery dock at cell (10, 10) │
└────────────────────────────────────────────┘
```

The AGV starts each episode at a random valid cell. It knows its current position. It knows its four possible moves. It knows nothing else — not the obstacle positions, not which aisles are congested, not how far it is from the dock.

The reward function is the only teacher:

| Event | Reward | What it teaches |
|---|---|---|
| Reach dock (10,10) | **+100** | The only thing that matters is getting there |
| Hit wall or obstacle | **−50** | Walls are expensive. Learn them fast. |
| Cross congestion zone | **−1 − 10 = −11** | Busy aisles cost time and throughput |
| Normal move | **−1** | Time is money. Don't wander. |

The negative step cost is not incidental — it is the pressure that produces the **shortest** path, not just any path that eventually reaches the dock.

---

## Episodes 1–100 — The Agent Knows Nothing

The agent begins with a Q-table of zeros: `Q(s, a) = 0` for all 121 states and 4 actions. Every entry says the same thing — *"I have no idea."*

Epsilon starts at 1.0. Every decision is random. The agent walks into walls. It wanders into congestion zones. It occasionally finds the goal by chance.

**Mean reward, Episodes 1–100: −980.9**

That number tells the full story. The agent is absorbing −50 wall penalties repeatedly, spending 200-step budgets without reaching the dock, accumulating hundreds of negative reward points per episode before the environment resets and starts over.

This is not failure. This is the cost of learning from scratch.

After each step, one update fires:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \left[ \underbrace{r + \gamma \cdot \max_{a'} Q(s',a')}_{\text{what we expected}} - Q(s,a) \right]$$

Where α = 0.1 (learning rate) and γ = 0.95 (discount factor — future rewards are worth 95 cents on the dollar). The bracket is the **TD error**: the gap between what the agent expected and what actually happened. Every collision writes a −50 into memory. Every goal arrival writes a +100. The Q-table is being carved by experience.

---

## Episodes 101–500 — The Curve Turns

Something shifts around episode 100. The Q-table has accumulated enough wall collisions that the agent has learned, imprecisely, which moves lead to punishment. Exploration is still dominant — epsilon has decayed from 1.0 to approximately 0.61 — but the agent is no longer purely random.

**Mean reward, Episodes 101–500: +29.2**

The agent is reaching the dock more often than not. Routes are longer than optimal — it hasn't learned to avoid congestion or route efficiently around the vertical obstacle wall — but the fundamental behavior has flipped: the agent is now net-positive.

By episode 312, the 200-episode rolling mean crosses zero for good. The agent has broken even. From this point forward, rewards accumulate.

---

## Episodes 500–5,000 — The Policy Takes Shape

By episode 500, epsilon has fallen to 0.082. The agent exploits its Q-table over 91% of the time. Random exploration becomes increasingly rare. What was carved roughly in the early episodes is now being refined.

**Mean reward, Episodes 501–1,000: +84.2**

The Q-table is converging toward its final form. Over 100 of the 121 cells have a clear dominant action — the entry that maximizes expected future reward is significantly larger than the alternatives. The policy is becoming deterministic.

The full arc of training, in five phases:

| Phase | Episodes | ε Range | Behavior | Mean Reward |
|---|---|---|---|---|
| Pure exploration | 1 – 100 | 1.00 → 0.61 | Random decisions, heavy wall penalties | −980.9 |
| Rapid learning | 101 – 500 | 0.61 → 0.08 | Pattern recognition, first reliable routes | +29.2 |
| Policy refinement | 501 – 1,000 | 0.08 → 0.05 | Route optimization, congestion avoidance | +84.2 |
| Exploitation dominant | 1,001 – 2,000 | 0.05 (floor) | Near-greedy behavior, stable navigation | +84.5 |
| Near-optimal routing | 2,001 – 5,000 | 0.05 (floor) | Full exploitation of learned policy | +85.4 |

The total improvement from first episodes to last: **−980.9 → +86.5** — a swing of over 1,060 reward points across 5,000 episodes of trial, consequence, and adaptation.

---

## Episode 5,000 — What the Agent Learned

The trained Q-table is a 121 × 4 matrix. Every entry is a number the agent earned the hard way.

**Policy distribution across 121 cells:**

| Best action | Cells | Interpretation |
|---|---|---|
| `down` | 57 | The dock is at row 10 — most cells need to move south |
| `right` | 45 | The dock is at col 10 — most cells need to move east |
| `up` | 18 | Cells below a wall segment that forces detour |
| `left` | 1 | One specific cell where left is the only valid escape |

The arrows collectively form a **flow field** across the factory floor: a river of decisions converging on (10,10), routing around the vertical wall at column 5 (rows 1–7) and the four scattered obstacle blocks.

**Q\* range across all states: 0.00 – 99.00** (mean: 50.22). High Q\* values cluster near the dock and along clear approach corridors. Low values mark dead-ends, cells adjacent to obstacles, and positions where the agent must take a long detour.

---

## Three Routes — The Agent in Action

The trained policy is tested from three start positions using the greedy strategy: at every cell, take the action with the highest Q-value.

**Scenario A — Corner to Corner** `start: (0,0)`

The maximum-distance run. The agent navigates from the top-left corner to the dock at (10,10), routing along the left side of the floor, cutting east below the vertical wall, and sweeping along the bottom row.

```
Path  : (0,0)→(0,1)→(1,1)→(2,1)→(3,1)→(4,1)→(4,2)→(5,2)→...→(10,10)
Steps : 20   |   Reward : 80.0   |   Goal reached : ✓
```

**Scenario B — Right Column Sprint** `start: (0,10)`

Starting at the top-right corner, the agent takes the clearest route toward the dock, staying close to column 10 and avoiding the vertical wall cluster.

```
Path  : (0,10)→(1,10)→(2,10)→(2,9)→(3,9)→(4,9)→(5,9)→(6,9)→...→(10,10)
Steps : 12   |   Reward : 88.0   |   Goal reached : ✓
```

**Scenario C — Congestion Navigation** `start: (5,0)`

Starting mid-left, the optimal route skirts the congestion zone rather than crossing it, adding extra column travel to avoid the −10 penalty on cells (4,6), (4,7), (4,8).

```
Path  : (5,0)→(5,1)→(5,2)→(5,3)→(5,4)→(6,4)→(7,4)→(8,4)→...→(10,10)
Steps : 15   |   Reward : 85.0   |   Goal reached : ✓
```

---

## Hyperparameter Sensitivity

The agent's behavior is shaped by three parameters. Each was tested while holding the others fixed:

**Learning rate (α)** — 2,000-episode runs:

| α | Last-200 Mean Reward | Verdict |
|---|---|---|
| 0.01 | 83.66 | Learns correctly but slowly — more episodes needed |
| 0.05 | 85.19 | Good |
| **0.10** | **85.61** | **Chosen — stable convergence** |
| 0.20 | 86.35 | Marginally better, less stable under noise |
| 0.50 | 89.61 | Fast but sensitive to stochastic episodes |

**Discount factor (γ)** — 2,000-episode runs:

| γ | Last-200 Mean Reward | Verdict |
|---|---|---|
| 0.70 | 86.17 | Myopic — adequate for nearby starts, poor for distant ones |
| 0.80 | 84.81 | Near-myopic |
| 0.90 | 84.00 | Slightly underplans in 11×11 horizon |
| **0.95** | **85.61** | **Chosen — correct horizon for 121-state environment** |
| 0.99 | 86.83 | Slightly better numerically, longer to converge |

γ = 0.95 means a reward 20 steps away (Scenario A) is still worth 95^20 ≈ 36% of its face value. For an 11×11 grid where the maximum optimal path is ~22 steps, the agent must plan far enough ahead — and 0.95 provides exactly that horizon.

---

## 🗂️ Repository Structure

```
QLearning_AGV/
├── 22_QLearning_AGV.ipynb   # Educational notebook (no outputs)
├── Data_AGV.csv             # Trained Q-table: 121 states × 9 columns
├── requirements.txt
└── README.md
```

**About `Data_AGV.csv`:** unlike every other CSV in this portfolio, this file is not training input — it is training output. Each row is a cell the agent has learned. Each Q-value column is a number earned through consequence.

> 📦 **Full Project Pack** — notebook with full outputs, presentation deck (PPTX + PDF),
> and `app.py` route simulator with interactive factory floor available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Colab (recommended):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/QLearning_AGV/blob/main/22_QLearning_AGV.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/QLearning_AGV.git
cd QLearning-AGV
pip install -r requirements.txt
jupyter notebook 22_QLearning_AGV.ipynb
```

**Requirements:** `numpy`, `pandas`, `matplotlib`

---

## 💡 Five Things the Agent Teaches Us

**1 — Consequence is a more honest teacher than labels.** Every labeled dataset in this portfolio carries human judgment embedded in it — someone decided what counts as a failure, an anomaly, a class. The Q-Learning agent receives only the physics of the environment. There is no label. There is only what happened next.

**2 — The Q-table is the model, and it is fully interpretable.** One hundred rows, four columns, one number per cell-action pair. You can read the policy directly: go to row 45, look at the four Q-values, pick the largest. No black box. No kernel trick. No embedding. A maintenance technician can verify the agent's decision with a spreadsheet.

**3 — Exploration decay is a design decision, not a detail.** The choice of ε_decay = 0.995 determines when the agent stops learning and starts performing. Too fast (0.99) and it commits to a suboptimal policy before seeing enough of the environment. Too slow (0.999) and it keeps exploring when it should be exploiting. Episode 299 — the break-even point — is a direct consequence of this choice.

**4 — The step cost is what makes the route optimal, not just feasible.** Without the −1 step penalty, the agent would reach the goal by any path — including a 50-step loop. The step cost creates pressure toward efficiency. In an industrial setting, this maps directly to cycle time, energy consumption, and throughput. The reward function is the engineering spec.

**5 — This is where the portfolio's paradigm changes.** Projects 01–21 recognized patterns in historical data. This agent generates new behavior in real time from a policy it built through 5,000 episodes of trial and error. The next chapter — policy optimization, model-based RL — extends that capability further. But the core shift happens here.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*

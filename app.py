"""
app.py — AGV Route Intelligence Dashboard
LozanoLsa · Project 22 · Tabular Q-Learning · 2026 · FREE PROJECT

Algorithm: Q-Learning (tabular, epsilon-greedy)
Domain: Factory Floor Navigation — AGV Optimal Routing
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AGV Route Intelligence · Q-Learning · LozanoLsa",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS — DISTILLATION STUDIO STYLE ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #ffffff;
    --surface:  #f8fafc;
    --card:     #f8fafc;
    --card2:    #f1f5f9;
    --border:   #e2e8f0;
    --border2:  #cbd5e1;
    --green:    #16a34a;
    --green2:   #4ade80;
    --green3:   #86efac;
    --danger:   #dc2626;
    --warn:     #d97706;
    --blue:     #2563eb;
    --text:     #0f172a;
    --text2:    #1e293b;
    --muted:    #64748b;
    --muted2:   #94a3b8;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] input { font-family: var(--fm) !important;
    font-size: 1rem !important; font-weight: 600 !important;
    color: var(--text) !important; background: #fff !important;
    border: 1px solid var(--border2) !important; border-radius: 4px !important; }
[data-testid="stNumberInput"] button { background: var(--card2) !important;
    border: 1px solid var(--border) !important; color: var(--text) !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--green) !important;
    border: 2px solid var(--green2) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--green) !important; }

[data-testid="stMetric"] { background: #fff !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--green) !important;
    padding: 0.9rem 1rem !important; border-radius: 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important;
    font-size: 0.6rem !important; text-transform: uppercase !important;
    letter-spacing: 0.16em !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important;
    font-size: 1.6rem !important; font-weight: 700 !important; color: var(--text) !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid var(--border) !important;
    background: var(--surface) !important; padding: 0 2.4rem !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important;
    font-size: 0.68rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: var(--muted) !important;
    padding: 0.7rem 1.2rem !important; border: none !important;
    background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--green) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important; font-weight: 600 !important; }
[data-testid="stTabsContent"] { padding: 1.6rem 2.4rem !important; }

[data-testid="stExpander"] { background: #fff !important;
    border: 1px solid var(--border) !important; border-radius: 4px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important;
    font-size: 0.75rem !important; color: var(--text) !important; font-weight: 600 !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 4px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important;
    font-size: 0.72rem !important; color: var(--text2) !important; background: #fff !important; }

[data-testid="stAlert"] { border-radius: 4px !important;
    font-family: var(--fm) !important; font-size: 0.75rem !important; }

hr { border-color: var(--border) !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important;
    font-size: 0.63rem !important; color: var(--muted) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

/* ── STUDIO COMPONENTS ── */
.studio-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #1a3a2a);
    padding: 2rem 2.4rem 1.6rem;
    margin-bottom: 0;
}
.studio-tag { font-family: var(--fm); font-size: 0.58rem; color: var(--green2);
    text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 6px; }
.studio-title { font-family: var(--fh); font-size: 2rem; font-weight: 800;
    color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.studio-subtitle { font-family: var(--fs); font-style: italic;
    font-size: 0.95rem; color: #94a3b8; margin-top: 6px; }
.studio-pill { display: inline-block; background: rgba(74,222,128,0.15);
    border: 1px solid rgba(74,222,128,0.4); color: #86efac;
    font-family: var(--fm); font-size: 0.6rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px;
    margin-right: 6px; margin-top: 12px; }
.studio-pill-free { display: inline-block; background: rgba(74,222,128,0.2);
    border: 1px solid rgba(74,222,128,0.5); color: #4ade80;
    font-family: var(--fm); font-size: 0.6rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px; border-radius: 20px;
    margin-right: 6px; margin-top: 12px; font-weight: 700; }

.section-label { font-family: var(--fh); font-size: 1.1rem; font-weight: 700;
    color: var(--text); margin-bottom: 6px; margin-top: 0; }
.section-desc { font-family: var(--fh); font-size: 0.82rem; color: var(--muted);
    margin-bottom: 1.2rem; line-height: 1.5; }
.input-card { background: #fff; border: 1px solid var(--border);
    border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.input-card-title { font-family: var(--fm); font-size: 0.65rem; font-weight: 600;
    color: var(--green); text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 10px; }
.result-card { background: #fff; border: 1px solid var(--border);
    border-radius: 8px; padding: 1.4rem 1.6rem; margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.lsa-footer { padding: 1rem 2.4rem; border-top: 1px solid var(--border);
    font-family: var(--fm); font-size: 0.58rem; color: var(--muted);
    letter-spacing: 0.1em; background: var(--surface); }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB LIGHT PALETTE ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#e2e8f0", "axes.labelcolor": "#0f172a",
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "text.color": "#0f172a", "grid.color": "#e2e8f0",
    "grid.linestyle": "--", "grid.alpha": 0.6,
    "legend.facecolor": "white", "legend.edgecolor": "#e2e8f0",
})

C_GREEN  = "#16a34a"; C_GREEN2 = "#4ade80"; C_GREEN3 = "#86efac"
C_DANGER = "#dc2626"; C_WARN   = "#d97706"; C_BLUE   = "#2563eb"
C_TEXT   = "#0f172a"; C_MUTED  = "#64748b"; C_BORD   = "#e2e8f0"
C_OBS    = "#ef4444"; C_CONG   = "#f59e0b"; C_GOAL   = "#16a34a"

# ─── ENVIRONMENT & MODEL ─────────────────────────────────────────────────────
class AGVGridWorld:
    def __init__(self, rows=11, cols=11):
        self.rows, self.cols = rows, cols
        self.n_states, self.n_actions = rows * cols, 4
        self.action_names = ["up", "down", "left", "right"]
        self.obstacles       = {(1,5),(2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(3,2),(3,8),(7,2),(8,8)}
        self.congestion_zones= {(4,6),(4,7),(4,8),(5,6),(5,7)}
        self.goal = (10, 10)
    def state_to_rc(self, s): return s // self.cols, s % self.cols
    def rc_to_state(self, r, c): return r * self.cols + c
    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.obstacles
    def sample_start(self):
        valid = [s for s in range(self.n_states)
                 if self.state_to_rc(s) != self.goal
                 and self.state_to_rc(s) not in self.obstacles]
        return int(np.random.choice(valid))
    def step(self, state, action):
        r, c = self.state_to_rc(state)
        dr = [-1,1,0,0]; dc = [0,0,-1,1]
        nr, nc = r + dr[action], c + dc[action]
        if not self.is_valid(nr, nc): return state, -50.0, False
        if (nr, nc) == self.goal:     return self.rc_to_state(nr, nc), 100.0, True
        reward = -1.0
        if (nr, nc) in self.congestion_zones: reward -= 10.0
        return self.rc_to_state(nr, nc), reward, False

def q_learning(env, episodes=5000, alpha=0.1, gamma=0.95,
               epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, max_steps=200):
    Q = np.zeros((env.n_states, env.n_actions)); rewards = []
    for _ in range(episodes):
        state = env.sample_start(); ep_r = 0.0
        for _ in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(Q[state, :]))
            ns, r, done = env.step(state, action)
            Q[state, action] += alpha * (r + gamma * np.max(Q[ns, :]) - Q[state, action])
            state = ns; ep_r += r
            if done: break
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(ep_r)
    return Q, rewards

def simulate_route(env, Q, start_rc, max_steps=100):
    state = env.rc_to_state(*start_rc); path = []; total_r = 0.0
    for _ in range(max_steps):
        r, c = env.state_to_rc(state); path.append((r, c))
        if (r, c) == env.goal: break
        action = int(np.argmax(Q[state, :])); ns, rew, done = env.step(state, action)
        total_r += rew; state = ns
        if done: path.append(env.state_to_rc(state)); break
    return path, total_r, env.state_to_rc(state) == env.goal, len(path) - 1

@st.cache_resource
def train():
    env = AGVGridWorld(); np.random.seed(42)
    Q, rewards = q_learning(env)
    return env, Q, rewards

env, Q, episode_rewards = train()

@st.cache_data
def build_qtable():
    for path in ["Data_AGV.csv", "22_QLearning_AGV/Data_AGV.csv"]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    records = []
    for s in range(env.n_states):
        r, c = env.state_to_rc(s)
        bi = int(np.argmax(Q[s, :]))
        records.append({"state":s, "row":r, "col":c,
                         "Q_up":round(Q[s,0],2), "Q_down":round(Q[s,1],2),
                         "Q_left":round(Q[s,2],2), "Q_right":round(Q[s,3],2),
                         "best_action":env.action_names[bi],
                         "best_Q":round(float(Q[s,bi]),2)})
    return pd.DataFrame(records)

df_q = build_qtable()

# ─── DRAW GRID (LIGHT THEME) ─────────────────────────────────────────────────
def draw_factory(env, path=None, agv_pos=None, title="", figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white"); ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, env.cols); ax.set_ylim(0, env.rows)
    # Grid lines
    for i in range(env.rows + 1): ax.axhline(i, color=C_BORD, lw=0.5)
    for j in range(env.cols + 1): ax.axvline(j, color=C_BORD, lw=0.5)
    # Obstacles
    for (r, c) in env.obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=C_OBS, alpha=0.75, zorder=2))
    # Congestion zones
    for (r, c) in env.congestion_zones:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color=C_CONG, alpha=0.30, zorder=2))
        ax.text(c+0.5, r+0.5, "⚠", ha="center", va="center", fontsize=8, zorder=3)
    # Goal
    gr, gc = env.goal
    ax.add_patch(plt.Rectangle((gc, gr), 1, 1, color=C_GOAL, alpha=0.80, zorder=2))
    ax.text(gc+0.5, gr+0.5, "G", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=4)
    # Route path
    if path and len(path) > 1:
        xs = [c+0.5 for (r, c) in path]; ys = [r+0.5 for (r, c) in path]
        ax.plot(xs, ys, color=C_GREEN, lw=2.5, zorder=5, alpha=0.7)
        # Start marker
        sr, sc = path[0]
        ax.add_patch(plt.Circle((sc+0.5, sr+0.5), 0.3, color=C_BLUE, zorder=6))
        ax.text(sc+0.5, sr+0.5, "S", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=7)
    # AGV current position (animation frame)
    if agv_pos is not None:
        ar, ac = agv_pos
        ax.add_patch(plt.Circle((ac+0.5, ar+0.5), 0.35, color=C_GREEN2, zorder=8,
                                 linewidth=2, edgecolor=C_GREEN))
        ax.text(ac+0.5, ar+0.5, "🤖", ha="center", va="center", fontsize=10, zorder=9)
    ax.invert_yaxis()
    ax.set_xticks([j+0.5 for j in range(env.cols)]); ax.set_xticklabels(range(env.cols), fontsize=7)
    ax.set_yticks([i+0.5 for i in range(env.rows)]); ax.set_yticklabels(range(env.rows), fontsize=7)
    if title: ax.set_title(title, fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
    fig.tight_layout()
    return fig

# ─── STUDIO HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="studio-header">
    <div class="studio-tag">LozanoLsa · Project 22 · Reinforcement Learning</div>
    <div class="studio-title">AGV Route Intelligence</div>
    <div class="studio-subtitle">Q-Learning teaches the AGV every optimal path through the factory floor — including around obstacles and congestion zones.</div>
    <div>
        <span class="studio-pill">Q-LEARNING</span>
        <span class="studio-pill">11×11 GRID</span>
        <span class="studio-pill">5,000 EPISODES</span>
        <span class="studio-pill">ROUTE ANIMATION</span>
        <span class="studio-pill">POLICY MAP</span>
        <span class="studio-pill-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
st.markdown('<div style="padding:1.2rem 2.4rem 0.8rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;">', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Grid Size",        "11 × 11",            "121 states · 4 actions")
k2.metric("Training Episodes","5,000",              "ε-greedy · decay 0.995")
k3.metric("Final Reward",     f"{np.mean(episode_rewards[-100:]):.0f}", "Last 100 ep average")
k4.metric("Convergence",      "~3,000 ep",          "Policy stable after ep 3k")
st.markdown('</div>', unsafe_allow_html=True)

# ─── MAIN AREA: SECTION 1 + SECTION 2 ────────────────────────────────────────
st.markdown('<div style="padding:1.6rem 2.4rem;">', unsafe_allow_html=True)

col_inp, col_vis = st.columns([1, 1.4])

with col_inp:
    st.markdown('<p class="section-label">1. Input Section</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Set the AGV start position and animation speed, then click <strong>Compute Route</strong> or <strong>Animate</strong> to watch the agent navigate.</p>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Start Position</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_r = st.number_input("Row (0–9)", min_value=0, max_value=9, value=0, step=1, key="sr")
    with c2:
        start_c = st.number_input("Col (0–9)", min_value=0, max_value=9, value=0, step=1, key="sc")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">Animation Speed</div>', unsafe_allow_html=True)
    speed = st.slider("Delay per step (seconds)", 0.05, 0.6, 0.18, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    compute = btn_col1.button("▶  Compute Route", use_container_width=True)
    animate = btn_col2.button("⚡  Animate AGV",  use_container_width=True)

    # Legend
    st.markdown(f"""
    <div style="background:#fff;border:1px solid {C_BORD};border-radius:8px;
                padding:1rem 1.2rem;margin-top:12px;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:8px;">Legend</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;
                    font-family:var(--fm);font-size:0.7rem;color:var(--text2);">
            <span><span style="color:{C_OBS};font-weight:700;">■</span> Obstacle</span>
            <span><span style="color:{C_GOAL};font-weight:700;">■</span> Goal (10,10)</span>
            <span><span style="color:{C_CONG};font-weight:700;">■</span> Congestion (−10r)</span>
            <span><span style="color:{C_GREEN};font-weight:700;">─</span> Optimal Route</span>
            <span><span style="color:{C_BLUE};font-weight:700;">●</span> Start (S)</span>
            <span><span style="color:{C_GREEN2};font-weight:700;">●</span> AGV Position</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_vis:
    st.markdown('<p class="section-label">2. Factory Floor Visualization</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">The schematic shows the 11×11 grid, obstacles, congestion zones, goal cell, and the AGV\'s computed optimal route.</p>', unsafe_allow_html=True)

    grid_placeholder = st.empty()

    # Check validity
    is_obstacle = (start_r, start_c) in env.obstacles
    is_goal     = (start_r, start_c) == env.goal

    if is_obstacle or is_goal:
        grid_placeholder.warning(f"Cell ({start_r},{start_c}) is {'an obstacle' if is_obstacle else 'the goal'}. Choose a different start.")
    else:
        path, reward, reached, steps = simulate_route(env, Q, (start_r, start_c))

        # Default: show computed route
        if not animate:
            fig = draw_factory(env, path=path, agv_pos=path[-1] if reached else path[-1],
                               title=f"Start ({start_r},{start_c}) → Goal (10,10)  ·  {steps} steps  ·  R={reward:.0f}")
            grid_placeholder.pyplot(fig, use_container_width=True); plt.close()

        # Animation
        if animate:
            for i, pos in enumerate(path):
                fig = draw_factory(env, path=path[:i+1], agv_pos=pos,
                                   title=f"AGV navigating... step {i+1}/{steps}")
                grid_placeholder.pyplot(fig, use_container_width=True); plt.close()
                time.sleep(speed)
            # Final frame — full route shown
            fig = draw_factory(env, path=path, agv_pos=path[-1],
                               title=f"Route complete — {steps} steps · R = {reward:.0f}")
            grid_placeholder.pyplot(fig, use_container_width=True); plt.close()

        # Result card
        eff = max(0, 1 - (steps / (env.rows + env.cols)))
        eff_color = C_GREEN if eff > 0.6 else C_WARN
        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:10px;">
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                            color:{C_TEXT};">Route Summary</div>
                <span style="background:{'#f0fdf4' if reached else '#fef2f2'};
                             color:{'#16a34a' if reached else '#dc2626'};
                             font-family:var(--fm);font-size:0.65rem;font-weight:700;
                             padding:4px 12px;border-radius:20px;
                             border:1px solid {'#bbf7d0' if reached else '#fecaca'};">
                    {'✓ GOAL REACHED' if reached else '✗ TIMEOUT'}
                </span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;
                        font-family:var(--fm);font-size:0.72rem;">
                <div>
                    <div style="color:var(--muted);font-size:0.58rem;text-transform:uppercase;
                                letter-spacing:.12em;">Steps</div>
                    <div style="color:{C_TEXT};font-size:1.4rem;font-weight:700;">{steps}</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.58rem;text-transform:uppercase;
                                letter-spacing:.12em;">Total Reward</div>
                    <div style="color:{C_TEXT};font-size:1.4rem;font-weight:700;">{reward:.0f}</div>
                </div>
                <div>
                    <div style="color:var(--muted);font-size:0.58rem;text-transform:uppercase;
                                letter-spacing:.12em;">Efficiency</div>
                    <div style="color:{eff_color};font-size:1.4rem;font-weight:700;">{eff:.0%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Efficiency gauge
        fig_g, ax_g = plt.subplots(figsize=(6, 0.9))
        fig_g.patch.set_facecolor("white"); ax_g.set_facecolor("white")
        ax_g.barh([0], [1.0], color="#f1f5f9", height=0.5, edgecolor=C_BORD)
        ax_g.barh([0], [eff], color=eff_color, height=0.5, alpha=0.9)
        ax_g.axvline(0.6, color=C_BORD, lw=1.2, ls="--")
        ax_g.set_xlim(0, 1); ax_g.set_yticks([])
        ax_g.set_xticks([0, 0.5, 0.6, 1.0])
        ax_g.set_xticklabels(["Inefficient", "", "Good", "Optimal"], fontsize=8, color=C_MUTED)
        for sp in ax_g.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True); plt.close()

st.markdown('</div>', unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "LEARNING CURVE", "POLICY MAP", "Q* HEATMAP", "ACTION PLAN"
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">3. Q-Learning Convergence</p>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Episode reward over 5,000 training episodes**")
        window = 100
        smooth = pd.Series(episode_rewards).rolling(window).mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(episode_rewards, alpha=0.2, color=C_GREEN, lw=0.6, label="Raw")
        ax.plot(smooth,          color=C_GREEN, lw=2.0,   label=f"{window}-ep moving avg")
        ax.axhline(np.mean(episode_rewards[-500:]), color=C_DANGER, ls="--", lw=1.5,
                   label=f"Final avg = {np.mean(episode_rewards[-500:]):.0f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Policy stabilises around episode 3,000. Beyond that, the agent consistently reaches the goal.")

    with cb:
        st.markdown("**Training statistics**")
        stats = pd.DataFrame({
            "Metric": ["Episodes", "α (learning rate)", "γ (discount)", "ε start", "ε end",
                       "ε decay", "Best episode reward", "Final avg (last 500)"],
            "Value":  ["5,000", "0.10", "0.95", "1.0", "0.05", "0.995",
                       f"{max(episode_rewards):.0f}", f"{np.mean(episode_rewards[-500:]):.0f}"],
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:3px solid #16a34a;
                    border-radius:4px;padding:10px 14px;margin-top:10px;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:#16a34a;
                        text-transform:uppercase;letter-spacing:.15em;margin-bottom:5px;">Key insight</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:#0f172a;line-height:1.6;">
                High γ=0.95 means the agent values future rewards almost as much as immediate ones —
                critical for navigating a long factory floor where the reward is only at the destination.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">4. Learned Policy — Optimal Action per Cell</p>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Arrow grid — best action per state**")
        action_sym = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("white"); ax.set_facecolor("#f8fafc")
        ax.set_xlim(0, env.cols); ax.set_ylim(0, env.rows)
        for i in range(env.rows + 1): ax.axhline(i, color=C_BORD, lw=0.5)
        for j in range(env.cols + 1): ax.axvline(j, color=C_BORD, lw=0.5)
        for _, rd in df_q.iterrows():
            r, c = int(rd["row"]), int(rd["col"])
            if (r, c) in env.obstacles:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color=C_OBS, alpha=0.70, zorder=2))
            elif (r, c) == env.goal:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color=C_GOAL, alpha=0.75, zorder=2))
                ax.text(c+0.5, r+0.5, "G", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold", zorder=4)
            else:
                sym  = action_sym.get(rd["best_action"], "?")
                col  = C_WARN if (r, c) in env.congestion_zones else C_GREEN
                alpha= 0.7  if (r, c) in env.congestion_zones else 1.0
                ax.text(c+0.5, r+0.5, sym, ha="center", va="center",
                        fontsize=13, color=col, fontweight="bold", zorder=3, alpha=alpha)
        ax.invert_yaxis()
        ax.set_xticks([j+0.5 for j in range(10)]); ax.set_xticklabels(range(10), fontsize=7)
        ax.set_yticks([i+0.5 for i in range(10)]); ax.set_yticklabels(range(10), fontsize=7)
        ax.set_title("Optimal Policy — Flow Field", fontsize=11, fontweight="bold", color=C_TEXT)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Arrows form a flow field converging on (10,10). Orange = congestion zone — agent still routes through if no better option.")

    with cb:
        st.markdown("**Q-Table sample — top 10 states by Q* value**")
        top_q = df_q.nlargest(10, "best_Q")[["state", "row", "col",
                                               "best_action", "best_Q"]].reset_index(drop=True)
        st.dataframe(top_q, use_container_width=True)
        st.caption("States closest to (10,10) with a clear path have the highest Q* — maximum expected future reward.")

        st.markdown("**Action distribution across all states**")
        action_dist = df_q[~df_q.apply(lambda r: (r["row"], r["col"]) in env.obstacles,
                                        axis=1)]["best_action"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        bar_c = [C_GREEN, C_BLUE, C_WARN, C_DANGER]
        bars  = ax2.barh(action_dist.index, action_dist.values,
                         color=bar_c[:len(action_dist)], alpha=0.85, edgecolor="white", height=0.5)
        for bar, v in zip(bars, action_dist.values):
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(v), va="center", fontsize=9, color=C_TEXT)
        ax2.set_xlabel("States with this best action")
        ax2.grid(True, axis="x", alpha=0.4)
        for sp in ax2.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">5. Q* State Value Heatmap</p>', unsafe_allow_html=True)
    ca, cb = st.columns([1.1, 1])
    with ca:
        grid_q = df_q.pivot(index="row", columns="col", values="best_Q")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        fig.patch.set_facecolor("white")
        im = ax.imshow(grid_q.values, cmap="YlGn", origin="upper", aspect="equal")
        plt.colorbar(im, ax=ax, label="Q* (max expected future reward)", shrink=0.85)
        for (r, c) in env.obstacles:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color=C_OBS, alpha=0.80, zorder=3))
        gr, gc = env.goal
        ax.text(gc, gr, "G", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", zorder=5)
        ax.set_title("Q* Heatmap — State Value Function", fontsize=11,
                     fontweight="bold", color=C_TEXT)
        ax.set_xlabel("Column"); ax.set_ylabel("Row")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORD)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption(f"Q* range: {df_q['best_Q'].min():.1f} to {df_q['best_Q'].max():.1f}. "
                   "Bright green = states with a clear path to the goal. Red = obstacles.")

    with cb:
        st.markdown("**Q* statistics**")
        q_stats = df_q[~df_q.apply(lambda r: (r["row"], r["col"]) in env.obstacles, axis=1)]
        for label, val in [
            ("Max Q* (closest to goal)", f"{q_stats['best_Q'].max():.2f}"),
            ("Mean Q* (free cells)",     f"{q_stats['best_Q'].mean():.2f}"),
            ("Min Q* (furthest/blocked)",f"{q_stats['best_Q'].min():.2f}"),
            ("Std deviation",            f"{q_stats['best_Q'].std():.2f}"),
        ]:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid {C_BORD};border-radius:4px;
                        padding:8px 12px;margin-bottom:6px;display:flex;
                        justify-content:space-between;align-items:center;">
                <span style="font-family:var(--fm);font-size:0.7rem;color:{C_MUTED};">{label}</span>
                <span style="font-family:var(--fm);font-size:0.82rem;font-weight:700;
                             color:{C_TEXT};">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:3px solid #16a34a;
                    border-radius:4px;padding:10px 14px;margin-top:12px;">
            <div style="font-family:var(--fm);font-size:0.6rem;color:#16a34a;
                        text-transform:uppercase;letter-spacing:.15em;margin-bottom:5px;">How to read this</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:#0f172a;line-height:1.6;">
                High Q* = the agent is confident it can reach the goal from this state with high reward.
                Low Q* = far away, blocked by obstacles, or in congestion zone.
                The heatmap IS the agent's understanding of the factory floor.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-label">6. Operational Use Cases</p>', unsafe_allow_html=True)

    use_cases = [
        {"title": "AGV Dynamic Rerouting", "color": C_GREEN,
         "scenario": "A machine goes offline, blocking a cell the AGV normally uses.",
         "rl_solution": "Add the blocked cell to `obstacles`. Retrain with 1,000 episodes from checkpoint. The Q-table adapts — no manual reprogramming of routes.",
         "value": "Eliminates manual route reprogramming. Adapts in minutes vs. hours of engineering time."},
        {"title": "Congestion-Aware Scheduling", "color": C_WARN,
         "scenario": "Certain aisles become congested during shift changes (08:00–09:00 and 16:00–17:00).",
         "rl_solution": "Time-dependent congestion zones. Train two Q-tables (peak/off-peak). Switch policy at shift boundaries.",
         "value": "Reduces transit time by routing around predictable congestion. Fewer collisions, higher throughput."},
        {"title": "Multi-Objective Routing", "color": C_BLUE,
         "scenario": "Minimize both travel time AND energy consumption (battery-critical AGV fleet).",
         "rl_solution": "Modify reward: R = −1 (step) − energy_cost(cell) + 100 (goal). Agent learns low-energy paths automatically.",
         "value": "Battery life extended. Fewer recharge cycles. Lower energy cost per delivery."},
        {"title": "Digital Twin Layout Validation", "color": C_DANGER,
         "scenario": "Before deploying a new factory layout, validate AGV routability.",
         "rl_solution": "Encode the new layout in AGVGridWorld. Train. If convergence fails, the layout has routing dead-ends.",
         "value": "Layout issues caught in simulation, not on the shop floor. Zero physical disruption."},
    ]

    for uc in use_cases:
        with st.expander(uc["title"], expanded=True):
            cl, cr = st.columns([2, 1])
            with cl:
                for lbl, text, bg, border in [
                    ("Industrial Scenario",  uc["scenario"],    "#f8fafc", C_BORD),
                    ("RL Solution",          uc["rl_solution"], "#f0fdf4", "#bbf7d0"),
                    ("Operational Value",    uc["value"],       "#eff6ff", "#bfdbfe"),
                ]:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border};
                                border-left:3px solid {uc['color']};border-radius:4px;
                                padding:8px 12px;margin-bottom:7px;">
                        <div style="font-family:var(--fm);font-size:0.58rem;color:{C_MUTED};
                                    text-transform:uppercase;letter-spacing:.15em;margin-bottom:4px;">{lbl}</div>
                        <div style="font-family:var(--fm);font-size:0.72rem;color:{C_TEXT};
                                    line-height:1.6;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            with cr:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid {C_BORD};
                            border-top:3px solid {uc['color']};border-radius:4px;
                            padding:1rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:{C_MUTED};
                                text-transform:uppercase;letter-spacing:.1em;">Type</div>
                    <div style="font-family:var(--fm);font-size:0.9rem;font-weight:700;
                                color:{uc['color']};margin-top:6px;">{uc['title']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:#fff;border:1px solid {C_BORD};border-radius:8px;
                padding:1.2rem 1.6rem;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
        <div style="font-family:var(--fh);font-size:1rem;font-weight:700;
                    color:{C_TEXT};margin-bottom:12px;">Q-Learning vs. Traditional Route Planning</div>
        <table style="width:100%;font-family:var(--fm);font-size:0.72rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid {C_BORD};background:#f8fafc;">
                <th style="padding:8px;text-align:left;color:{C_MUTED};">Criterion</th>
                <th style="padding:8px;color:{C_MUTED};">Traditional (Fixed)</th>
                <th style="padding:8px;color:{C_MUTED};">Q-Learning (Adaptive)</th>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_TEXT};">New obstacle</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Manual reprogramming</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Retrain in minutes</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_TEXT};">Congestion handling</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Fixed avoidance rules</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Learned penalty avoidance</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_TEXT};">Multi-objective</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Hard to encode</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Modify reward function</td>
            </tr>
            <tr>
                <td style="padding:8px;color:{C_TEXT};">Layout validation</td>
                <td style="padding:8px;text-align:center;color:{C_DANGER};">Physical testing required</td>
                <td style="padding:8px;text-align:center;color:{C_GREEN};">Simulation first</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · AGV Route Intelligence · Project 22 · v2.0
    &nbsp;·&nbsp;
    <a href="https://github.com/LozanoLsa" style="color:{C_GREEN};text-decoration:none;">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://lozanolsa.gumroad.com" style="color:{C_GREEN};text-decoration:none;">Gumroad</a>
</div>
""", unsafe_allow_html=True)

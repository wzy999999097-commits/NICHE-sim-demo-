# Family ABM — 家庭社会小生境智能体建模框架

一个模块化、可复用的**智能体建模（Agent-Based Model）框架**，用于模拟家庭社会小生境（Social Micro-Niches）。集成兰彻斯特型ODE拟合、交互式Web仪表板，面向计算社会学与交叉科学研究。

---

## 核心功能

- **多层次仿真** — 家庭包含多个成员智能体，每人拥有独立的人格、年龄轨迹、教育、收入、健康、情绪状态
- **交互式仪表板** — FastAPI + Plotly.js，内置中英文双语，在浏览器中完成仿真配置、运行、拟合、可视化全流程
- **ODE 拟合** — 7种内置社会动力学模型（竞争、影响、幸福-压力平衡等），支持多起点鲁棒优化和差分进化全局搜索
- **机器学习接口** — 状态记录 → DataFrame → 特征提取，可直接对接 sklearn / PyTorch 工作流
- **小生境理论** — 将智能体映射到四维社会空间（经济 / 文化 / 社会 / 情感），计算生境距离与重叠度
- **参数可调** — 18个动力学参数（学习速率、收入曲线、压力敏感度等）在仪表板中实时调整

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install -e .

# 2. 启动 Web 仪表板（自动打开浏览器）
python -m family_abm.web

# 3. 或从命令行运行示例
python examples/simple_family.py        # 基础仿真
python examples/ml_ready.py             # ML 特征提取
python examples/fitting_viz_demo.py     # 拟合 + 可视化
```

---

## 使用指南

### Web 仪表板

```bash
python -m family_abm.web
# → http://127.0.0.1:8520
```

| 标签页 | 功能 |
|--------|------|
| **Setup（设置）** | 配置家庭与成员、仿真步数、以及 18 个动力学参数。点击 *Run Simulation* 运行。 |
| **Charts（图表）** | 交互式时间序列图 + 社会生境空间散点图。 |
| **Fitting（拟合）** | 选择 ODE 模型，拟合仿真数据，查看 R² 与参数估计值。 |
| **Network（网络）** | 家庭关系网络图（情感、信任、冲突可视化）。|

右上角 ⚙ 按钮切换 **English / 简体中文**。
<img width="2531" height="1264" alt="image" src="https://github.com/user-attachments/assets/d9c9b193-79bd-4528-a2a1-df1b5b90dbc5" />

### Python API

```python
from family_abm import (
    Environment, Simulation, Scheduler,
    FamilyMember, Household, StateRecorder,
    make_fitter, wellbeing_balance, solve_model,
    plot_timeseries, plot_fit_diagnostics,
)

# 构建仿真
env = Environment()
hh = Household(name="张")
env.add_agent(hh)
hh.add_member(FamilyMember(name="父亲", age=35, role_name="parent"))
hh.add_member(FamilyMember(name="儿子", age=8, role_name="child"))

# 运行
recorder = StateRecorder()
sim = Simulation(env, scheduler=Scheduler("sequential"))
sim.add_recorder(recorder)
sim.run(120)

df = recorder.to_dataframe()

# 拟合幸福-压力模型
fitter = make_fitter("wellbeing")
fitter.fit_robust(df)
print(fitter.summary())

# 可视化
plot_timeseries(df)
plot_fit_diagnostics(df, fitter)
```

---

## 项目结构

```
family_abm/
├── core/              # 核心引擎：Agent, Environment, Scheduler, Simulation
├── family/            # 家庭模型：FamilyMember, Household, Relationship, Roles
├── niche/             # 生境模型：MicroNiche, ResourceBundle, InfluenceRelation
├── ml/                # ML 集成：StateRecorder, FeatureExtractor
├── fitting/           # 7种兰彻斯特型 ODE 模型 + ABMFitter
│   ├── lanchester.py  #   competition / influence / wellbeing / logistic / LV / resource
│   └── fitter.py      #   ABMFitter（fit_robust, fit_global）、make_fitter
├── viz/               # 静态可视化（matplotlib）：时间序列、相图、生境、网络、拟合诊断
├── web/               # FastAPI 仪表板 + Plotly.js
│   ├── __main__.py    #   启动入口：python -m family_abm.web
│   ├── app.py         #   REST API（run, data, fit, niche, network）
│   ├── templates/     #   HTML（data-i18n 双语支持）
│   └── static/        #   CSS + JS
└── __init__.py
```

---

## 智能体动力学

每个 `FamilyMember` 经历 **学龄前 → 儿童 → 学生 → 成人 → 老年** 五个生命阶段。内部状态每步更新：

| 变量 | 驱动机制 |
|------|---------|
| **教育** | 年龄依赖的学习速率，饱和于 1.0 |
| **收入** | `基础值 × (1 + 教育加成 × 受教育水平) × 年龄曲线 × 角色乘数` |
| **健康** | 基础衰减 + 年龄加速衰减，受教育水平减缓 |
| **压力** | 基础压力 + 工作负荷，受神经质人格放大，自然衰减 |
| **幸福** | 均值回归：`目标值 = 基底 + 健康权重×健康 + 收入权重×收入 + 教育权重×教育 − 压力惩罚×压力` |
| **精力** | 简单的恢复-消耗循环 |

所有速率、权重、噪声强度均可从仪表板调节。

---

## ODE 模型（兰彻斯特型）

| 模型 | 状态变量 | 参数 | 社会类比 |
|------|---------|------|---------|
| `square_law` | R₁, R₂ | α, β, ε₁, ε₂ | 群体竞争 |
| `linear_law` | R₁, R₂ | α, β | 交互驱动竞争 |
| `influence` | O₁, O₂ | γ, δ, T₁, T₂ | 意见/行为影响 |
| `wellbeing` | H, S | p, q, r, s, I | 幸福-压力平衡 |
| `logistic` | P | r, K | 单群体增长 |
| `lotka_volterra` | P, Q | a, b, c, d | 工作 vs 家庭时间 |
| `resource_competition` | R₁, R₂ | r₁, k₁, r₂, k₂, α, β | 逻辑斯蒂 + 兰彻斯特 |

拟合使用 `scipy.optimize.minimize`，支持多起点鲁棒优化（`fit_robust`）和差分进化全局搜索（`fit_global`）。

---

## 依赖

```
numpy, pandas, scipy, matplotlib, networkx
fastapi, uvicorn, jinja2
```

---


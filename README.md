# OpenClaw Skills

OpenClaw 投资分析 Skills

## Skills

### 一级投资

#### ai-funding-tracker

硅谷AI融资动态追踪 - 日报/周报/月报，洞察一级市场趋势

**功能：**
- 日报模式（daily）：当日融资事件速览
- 周报模式（weekly）：本周融资汇总与赛道深度分析
- 月报模式（monthly）：月度趋势洞察与行业判断

**使用：**
```bash
/ai-funding-tracker weekly
```

**重点关注：**
- 赛道分类：AI Infra、AI Agent、AI Coding、AI Robotics、B端应用、C端应用
- 轮次重点：A/B轮（共识形成阶段）
- 深度分析：Agent和Coding赛道演进、Market Map、大厂动作、投资逻辑演变
- 创始人背景：学术/大厂/连续创业/行业老兵
- 输出格式：Telegram消息

---

### 二级投资

#### earnings-decoder

买方视角财报解读 - 从价格反推假设，用数据验证，输出仓位决策

**功能：**
- 财报解读模式：财报发布后的深度分析
- 盘前模式（--pre-earnings）：财报前的策略分析

## 目录结构

```
plugins/
├── 一级投资/
│   └── skills/
│       └── ai-funding-tracker/
│           └── SKILL.md
└── earnings-decoder/
    └── skills/
        └── earnings-decoder/
            ├── SKILL.md
            ├── references/
            └── assets/
```

## 安装

```bash
# 复制到 OpenClaw skills 目录
cp -r plugins/一级投资/skills/ai-funding-tracker ~/.openclaw/skills/
cp -r plugins/earnings-decoder/skills/earnings-decoder ~/.openclaw/skills/
```

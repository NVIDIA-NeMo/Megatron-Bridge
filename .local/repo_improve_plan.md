# 总体路线

先做这条链路：

**入口清楚 → 中间可推进 → 末端可交接 → 最后再看全局优化**

# Phase 0：先定目标，不写代码

这一步很重要，不然 agent 很容易做散。这里先把 `Megatron-Hub` 当前最值得优先优化的问题写死，后面的 phase 都围绕这三件事推进。

## Step 0.1 定你现在最痛的 3 个问题

先只写三条，不要超过三条。对当前仓库，建议第一阶段只服务这 3 个问题：

* **review 完成后的 PR merge 太慢**
  不是技术判断卡住，而是卡在 CI 状态分散、base 过期、conflict、缺少明确 owner 这些机械问题上，导致已经有 reviewer 结论的 PR 仍然要来回推进。
* **upstream / submodule 变更容易把主线或下游打坏**
  尤其是 `Megatron-LM` / mcore bump、nightly 变更、兼容性回归，问题往往不是在变更发生时被显式记录，而是在后续 PR、功能测试或下游项目里才暴露。
* **handoff 和 follow-up 依赖人脑，状态容易丢**
  PR / issue 的 next step、blocking reason、owner、是否需要后续修复，经常只存在于评论、聊天或本地草稿里，不利于 oncall、last-mile agent 和后来接手的人持续推进。

这三条就是第一阶段 agent 的服务对象。先把这三条做好，比一开始就追求全能 agent 更重要。

## Step 0.2 定一个 north star metric

建议先固定 2 个主指标，再加 1 个 guardrail：

* **approved PR → merged 的中位时间**
* **需要 handoff / follow-up 但没有明确 owner 和 next step 的 PR / issue 数量**
* **guardrail：因 upstream / bump 引入并在 `main` 持续超过 24 小时的 breakage 数量**

以后每做完一阶段，就回头看这 3 个数字有没有改善。只要指标不动，就说明 agent 还没有真正减轻维护负担。

## Step 0.3 定 agent 原则

先把边界写死，避免 agent 越做越重：

* agent 可以 **建议** label、owner、next step、blocking reason
* agent 可以做 **机械动作**，比如汇总 checks、判断 base 是否过期、检查 conflict、起草 handoff / follow-up comment、整理 daily report
* agent 的结论要尽量落到 **可见载体** 上，比如 PR comment、issue、label、project view 或 repo 内文档，而不是只停留在聊天里
* agent 不做最终技术判断
* agent 不自动 merge 高风险 PR
* agent 不决定是否接受 design、breaking change 或大范围 refactor
* agent 不在没有明确 owner 的情况下默默接管长期 follow-up

这能防止后面边界越来越糊，也能保证后续 phase 的自动化是在可控范围内展开。

---

# Phase 1：先把 repo 基础结构立起来

这一阶段先不追求“聪明”，先追求“可被 agent 理解”。

## Step 1.1 先整理 label system

你不用一次搞很多，先上最核心的一组。

### 类型

* `bug`
* `feature`
* `support`
* `docs`
* `ci`

### 状态

* `needs-triage`
* `blocked`
* `needs-author`
* `ready-to-merge`
* `follow-up`

### 风险 / 特征

* `high-complexity`
* `breaking-change`

### area

* `area:runtime`
* `area:checkpoint`
* `area:config`
* `area:ci`
* `area:docs`

先有这些就够了。

## Step 1.2 建 PR template

重点不是写漂亮，而是让 agent 有结构化输入。

最少保留这几项：

* this PR changes
* why
* risk level
* tests added / affected
* docs updated?
* follow-up needed?

最后这个 `follow-up needed?` 很关键。

## Step 1.3 建 issue template

至少让新 issue 带这些：

* problem
* repro
* expected behavior
* environment
* logs

这样后面 triage agent 才不至于一直面对自由文本垃圾输入。

## Step 1.4 建 4 个固定视图

你自己和 oncaller 每天只看这 4 个：

* `needs-triage`
* `ready-to-merge`
* `follow-up OR blocked`
* `high-complexity`

做到这一步，其实 repo 已经会明显清爽很多。

---

# Phase 2：把你现在的 last mile agent 做“产品化”

你已经有雏形了，这一步是把它从一个脚本，变成一个清晰模块。

## Step 2.1 明确它的输入

先写清楚它吃什么信息：

* PR review state
* required checks
* base 是否过期
* 是否有 conflict
* changed files
* labels
* 最近 comment / request changes

## Step 2.2 明确它的输出状态

不要输出自由文本，先收敛成几个固定状态：

* `ready`
* `needs-rebase`
* `needs-retest`
* `blocked-by-conflict`
* `blocked-by-review`
* `blocked-by-missing-check`

这样后面 handoff 和 merge queue 都能直接复用。

## Step 2.3 明确它能做的动作

建议先只允许这些：

* 触发 CI rerun
* 提示需要 rebase
* 检查 branch freshness
* 生成 merge-ready summary
* 给 PR 加标签

如果你们流程允许，再逐步加：

* 自动 update branch
* 自动生成 handoff note

## Step 2.4 给它加 comment 模板

别让它每次说得不一样。

比如统一输出：

* 当前状态
* 为什么卡住
* 建议下一步
* 是否建议传给下周 oncaller

这样团队更容易接受。

## Step 2.5 先只覆盖一类 PR

不要全仓库一起上。

先挑：

* 已经 approved 的 PR
* 且不是 breaking-change
* 且不是高风险 area

跑一两周看看效果。

---

# Phase 3：补上入口和交接，形成闭环

你现在已经有 last mile，下一步最值得补的是前后两端。

---

## Part A：做 Triage Agent

### Step 3.1 先不自动回复，只做内部建议

它先做这些：

* 识别 issue / PR 类型
* 建议 labels
* 建议 area
* 判断是否缺信息
* 判断是否像 support 而不是 bug

先不要自动发 comment，避免误伤。

### Step 3.2 再加 label suggestion 或自动打简单 label

先自动的只建议这些低风险 label：

* `needs-triage`
* `support`
* `docs`
* `ci`
* `area:*`

不要一开始就自动打 priority 或 closing judgement。

### Step 3.3 最后再加回复草稿

等你确认准确率可以，再让它生成：

* 请求补 repro
* 建议补 logs
* 指向 FAQ / docs
* 建议转 discussion

---

## Part B：做 Handoff Agent

这是你现在很需要的。

### Step 3.4 先定义 handoff source

Handoff agent 先只看这些对象：

* `follow-up`
* `blocked`
* `needs-author`
* `ready-to-merge`

不要一开始扫全 repo。

### Step 3.5 统一 handoff 模板

固定成 4 段就够了：

* closed this week
* still in progress
* needs follow-up next week
* risks / blockers

### Step 3.6 每周固定时间跑一次

比如周五下午或周一早上。
先别追求实时，先有稳定节奏。

### Step 3.7 要求每个 carry-over item 有 3 个字段

每个传递项必须有：

* current status
* next action
* owner / who to poke

没有这三项的 handoff 基本没用。

---

# Phase 4：开始处理你说的 conflict 和 CI turn-around

这是在闭环形成后再做，不然会太散。

## Step 4.1 做 Merge Queue Agent，但先只做建议，不执行

它先输出：

* 哪些 PR ready
* 建议 merge 顺序
* 哪些现在 rerun CI 是浪费
* 哪些有 high-complexity

先给 oncaller 参考，不自动操作。

## Step 4.2 给高冲突 PR 打标

可以按 changed files 规则先做一版：

* 改 shared config
* 改 CI yaml
* 改 common mapping / registry
* 改 frequently touched core files

自动加 `high-complexity`。

## Step 4.3 建简单 merge policy

最小版本就够了：

* ready PR 不直接各自 merge
* 由 oncaller / sheriff 看 queue
* 高冲突 PR 优先
* base 过期的先更新再决定要不要重跑 full CI

## Step 4.4 区分 rebase-only 和 real code changes

这是减少 CI 浪费的关键基础。

哪怕一开始做不到完全自动，也要让系统能识别：

* 只是跟 main 对齐
* 真改了代码逻辑

这样以后才能接 test routing。

---

# Phase 5：开始看 repo 系统性健康，而不是只救火

这一阶段是你从“维护者”走向“repo lead”的关键。

## Step 5.1 做 Repo Health Agent

先每周只输出 5 条，不要一大堆图表。

建议包括：

* 最 flaky 的 tests
* 最常产生 conflict 的 files / areas
* approve-to-merge 时间最长的 PR
* 长时间没人动但重要的 issue
* docs drift / missing tests 的热点

## Step 5.2 做 Release Notes Agent

自动从 merged PR 里提炼：

* feature
* bugfix
* breaking change
* docs follow-up

这个很适合开源 repo，也能倒逼 PR 规范化。

## Step 5.3 再考虑 Test Routing Agent

只有在你们 CI 分层已经比较明确时再做。
不然它很容易变成另一个混乱来源。

---

# 你接下来最推荐的实际顺序

如果你问我“从明天开始怎么做”，我会建议这个顺序：

## Week 1

做 repo 基础：

* label system
* PR template
* issue template
* 4 个 saved views

## Week 2

把 last mile agent 正式化：

* 固定输入
* 固定输出状态
* 固定 comment 模板
* 先只管 approved PR

## Week 3

做 handoff agent：

* 固定 handoff source
* 固定 weekly template
* 开始每周生成 summary

## Week 4

做 triage agent：

* 先做 label / area 建议
* 暂不自动回复

## Week 5

做 merge queue suggestion：

* 不自动 merge
* 先只给排序建议和 risk warning

## Week 6+

开始 repo health 和 release notes

---

# 你每一步都应该交付什么

为了避免“做了很多但没落地”，每一阶段都要有一个可见交付物。

## Phase 1 交付物

* label 文档
* PR / issue template
* 4 个 GitHub views

## Phase 2 交付物

* last mile agent spec
* 状态枚举
* comment 模板
* 一周试运行结果

## Phase 3 交付物

* weekly handoff markdown 模板
* handoff agent 输出样例
* triage label 建议样例

## Phase 4 交付物

* merge queue recommendation 模板
* high-complexity 规则
* 简版 merge policy

## Phase 5 交付物

* weekly repo health digest
* release notes draft 模板

---

# 一个很实用的原则

每次只做一种能力升级：

* 先 **看见问题**
* 再 **标记问题**
* 再 **建议动作**
* 最后才 **自动执行**

比如 last mile agent：

1. 先看见 PR 卡在哪
2. 再打标签
3. 再建议 rerun / rebase
4. 最后才自动帮你触发

这样风险最低。

---

# 给你的最小起步版

如果你现在只想马上开始，那就先做这 3 件：

### 第一步

整理 label + PR template

### 第二步

把 last mile agent 固定成 6 个状态输出

### 第三步

做 weekly handoff template，让所有 `follow-up / blocked` 项都能自动进 handoff

做到这里，你就已经不是“乱管理”，而是在搭一个真正的 repo operating system 了。

下一条我可以直接帮你写：
**第一阶段落地清单**，按你可以直接贴到 GitHub repo 里的 markdown 形式给你。

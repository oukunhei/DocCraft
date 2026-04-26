# DocCraft

DocCraft 是一个基于 LangGraph 的动态工作流，用于从项目论文 + Word 模板自动生成高质量技术报告。

## 架构特点

```
preprocess_project_paper
       ↓
extract_template_requirements
       ↓
   plan_outline  ← LLM 生成全局大纲、术语表、跨槽位依赖
       ↓
   [Send] ──并行派发所有 slot ──→ slot_subgraph
                                    (每个 slot 独立子图)
                                    gather_evidence
                                          ↓
                                    generate_draft (结构化输出)
                                          ↓
                                    self_review (结构化输出)
                                          ↓
                             ┌─[confidence<0.7]─→ refine_draft ─┐
                             └────────(loop, max 3)─────────────┘
                                          ↓
                                    finalize_slot
       ↓
   aggregate_slots  ← 收集所有 slot_output
       ↓
   consistency_check  ← LLM 跨槽位一致性审查
       ↓
   build_report_docx  ← 模板样式继承，不再是等线！
```

### LangGraph 深度化
- **并行子图**：使用 `langgraph.types.Send` 将每个 slot 独立派发到子图，所有槽位并行生成
- **迭代循环**：子图内部实现 `generate → review → refine` 闭环，低置信度槽位自动改写（最多 3 轮）
- **条件路由**：`self_review` 后根据置信度决定是继续 refine 还是 finalize
- **状态持久化**：接入 `MemorySaver`，支持中断恢复与可重复执行
- **一致性审查**：全槽位生成后增加跨槽位一致性检查（术语/数据/逻辑）

### 生成质量提升
- **语义检索**：基于 FAISS + OpenAI Embeddings 的混合检索（语义 + 关键词），替代原有粗糙的关键词计数
- **大纲规划**：生成前由 LLM 输出全局写作计划，统一术语、规范跨槽位数据引用
- **结构化输出**：所有 LLM 节点使用 `with_structured_output(Pydantic)`，彻底消除 JSON 正则解析
- **引用可溯源**：证据召回时标注语义/关键词来源，生成要求标注证据编号

### Word 格式规范引擎
- **TemplateStyleProfile**：自动分析模板中每个占位符段落的字体、字号、加粗、颜色、对齐、行距、段前段后
- **格式保留替换**：`para.text = ...` 改为 `para.clear() + add_run()` + 显式字体克隆，彻底解决等线退化问题
- **样式克隆插入**：新插入的段落完全继承模板对应段落的样式（含东亚字体 `w:eastAsia`）
- **富文本支持**：生成内容中的 `**bold**` / `*italic*` 自动解析为 Word 加粗/斜体 run
- **字体 Fallback 链**：宋体/黑体/微软雅黑等中文字体优先匹配，防止退化

## Quick Start

### 1) 安装 uv

参考官方文档安装：https://docs.astral.sh/uv/

### 2) 配置环境变量

复制 `.env.example` 为 `.env`，填写 API 配置：

- `REASONING_API_KEY`
- `REASONING_URL` 或 `REASONING_BASE_URL`
- `TOOL_API_KEY`
- `TOOL_URL` 或 `TOOL_BASE_URL`
- 可选：`JINA_API_KEY`

### 3) 一键运行

```bash
uv sync
uv run python flow.py
```

建议首次运行先执行 `uv sync`，后续可直接 `uv run python flow.py`。

## 常用运行方式

默认输入（项目根目录）：

```bash
uv run python flow.py
```

指定输入文件：

```bash
uv run python flow.py \
    --source-file ./project_doc.pdf \
    --template-file ./template.docx
```

指定输出目录和 run id：

```bash
uv run python flow.py \
    --intermediate-dir ./artifacts/intermediate \
    --run-id 20260408T100000Z
```

调整槽位迭代次数（默认 3 轮）：

```bash
uv run python flow.py --max-slot-iterations 5
```

## 路径输入规则（已处理）

`flow.py` 现在对 `--source-file` 和 `--template-file` 做了容错解析：

1. 绝对路径（含 `~`）
2. 相对当前工作目录
3. 相对项目根目录

`--intermediate-dir` 若传相对路径，会自动解析到项目根目录下。

这意味着你从项目根目录外执行命令时，路径也更不容易出错。

## 输出位置

- 中间产物：`artifacts/intermediate/node1/{run_id}/`
- 中间产物：`artifacts/intermediate/node2/{run_id}/`
- 最终报告：`artifacts/final/report_filled_{run_id}.docx`
- 复核清单：`artifacts/final/report_review_checklist_{run_id}.docx`

## 终端日志说明

- node1：预处理论文、构建语义检索索引
- node2：解析模板 slot、分析模板样式
- plan_outline：LLM 生成全局大纲
- slot_subgraph：逐槽位并行生成（会打印迭代次数与置信度）
- aggregate_slots：聚合结果与统计
- consistency_check：跨槽位一致性评分
- build_report_docx：模板样式继承的 docx 生成

## 技术栈

| 模块 | 技术 |
|------|------|
| 工作流编排 | LangGraph + MemorySaver |
| 并行派发 | `langgraph.types.Send` + 子图 |
| 语义检索 | FAISS + OpenAI Embeddings |
| LLM 调用 | LangChain `ChatOpenAI.with_structured_output` |
| Word 处理 | python-docx + 自定义样式引擎 |
| 结构化数据 | Pydantic v2 |

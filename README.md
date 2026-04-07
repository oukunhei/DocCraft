# DocCraft

DocCraft 是一个 5 节点工作流：

1. node1 预处理项目论文
2. node2 提取模板槽位
3. node3 检索外部信息
4. node4 双 Agent 交叉分析
5. node5 生成填充后的报告 docx

本项目已按 uv 方式包装，可直接用 uv run 命令启动。

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

## 路径输入规则（已处理）

`flow.py` 现在对 `--source-file` 和 `--template-file` 做了容错解析：

1. 绝对路径（含 `~`）
2. 相对当前工作目录
3. 相对项目根目录

`--intermediate-dir` 若传相对路径，会自动解析到项目根目录下。

这意味着你从项目根目录外执行命令时，路径也更不容易出错。

## 输出位置

- 中间产物：`artifacts/intermediate/node{1..4}/{run_id}/`
- 最终报告：`artifacts/final/report_filled_{run_id}.docx`
- 复核清单：`artifacts/final/report_review_checklist_{run_id}.docx`

## 终端日志说明

- node1 / node2：会打印 `start` / `done`（以及产物目录）
- node3 / node4：会打印检索、争议和裁决过程

如果 node2 走了回退逻辑，可查看：

- `artifacts/intermediate/node2/{run_id}/meta.json`

其中会标记 `used_fallback` 与 `fallback_reason`。
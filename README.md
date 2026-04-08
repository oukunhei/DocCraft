# DocCraft

DocCraft 是一个 5 节点工作流：

1. node1 预处理项目论文
2. node2 提取模板槽位
3. node3 逐槽位生成草稿（Agent A）
4. node4 逐槽位复查并记录问题（Agent B，仅标注不改写）
5. node5 生成填充后的报告 docx

引用与参考文献策略：

- 非参考文献槽位：仅允许文内引用格式（如 `(Author, Year)` 或 `[1]`），严格忠于原论文证据。
- 参考文献槽位：允许基于论文名或 DOI 进行联网检索补全，最终在文末“参考文献”章节输出APA格式条目。

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

- 中间产物：`artifacts/intermediate/node1/{run_id}/`
- 中间产物：`artifacts/intermediate/node2/{run_id}/`
- 中间产物：`artifacts/intermediate/node4a/{run_id}/`
- 中间产物：`artifacts/intermediate/node5/{run_id}/`
- 最终报告：`artifacts/final/report_filled_{run_id}.docx`
- 复核清单：`artifacts/final/report_review_checklist_{run_id}.docx`

## 终端日志说明

- node1 / node2：会打印 `start` / `done`（以及产物目录）
- node3：会打印逐槽位生成进度（参考文献槽位会记录联网检索使用情况）
- node4：会打印逐槽位复查进度与分歧统计
- node5：会打印文档写入与输出信息

如果 node2 走了回退逻辑，可查看：

- `artifacts/intermediate/node2/{run_id}/meta.json`

其中会标记 `used_fallback` 与 `fallback_reason`。
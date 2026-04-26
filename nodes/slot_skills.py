from __future__ import annotations

import re
from typing import Any, Dict

# 统一槽位类别，供生成节点与复查节点复用。
SLOT_TYPE_REFERENCE = "reference"
SLOT_TYPE_BACKGROUND = "background"
SLOT_TYPE_RELATED = "related_work"
SLOT_TYPE_TECHNICAL = "technical_solution"
SLOT_TYPE_IMPLEMENTATION = "system_implementation"
SLOT_TYPE_TESTING = "testing_analysis"
SLOT_TYPE_INNOVATION = "innovation"
SLOT_TYPE_GENERAL = "general"

_SKILL_RULES: Dict[str, Dict[str, str]] = {
    SLOT_TYPE_REFERENCE: {
        "generation": (
            "【参考文献槽位 - 高精度生成规则】"
            "1. 内容范围：仅输出与项目核心技术、背景调研、对比实验直接相关的文献。"
            "2. 格式约束：强制使用APA 7th风格。格式模板：作者姓, 名首字母. (年份). *文章标题*. 期刊/会议名称, 卷(期), 页码. DOI/URL。\n"
            "3. 真实性校验：严禁编造DOI、卷期号或虚构作者。若证据材料中未提供完整书目信息，必须标注 '[需补充: 缺失作者/年份/DOI]'。\n"
            "4. 引用映射：在文末参考文献列表生成后，需在括号内注明该文献对应的正文引用上下文（如 `[用于支撑引言痛点数据]`）。"
        ),
        "review": (
            "【参考文献槽位 - 严格复查规则】\n"
            "1. 格式校验：检查标点符号（英文句点、斜体）、作者名缩写规范、DOI超链接格式。\n"
            "2. 文内一致性：逐一核对正文中的 `[n]` 索引是否与文末条目完全对应，避免漏引或错位。\n"
            "3. 时效性检查：若为技术方案对比，检查近3年文献占比是否合理（除经典奠基性文献外）。\n"
            "4. 异常反馈格式：如发现问题，反馈格式为 `[REF-ERR] 条目X: 具体错误描述及修正建议`。"
        ),
    },
    SLOT_TYPE_BACKGROUND: {
        "generation": (
            "【问题背景槽位 - 高精度生成规则】\n"
            "1. 三层递进结构：\n"
            "   - 宏观层：引用权威行业报告或政策数据（如Gartner、Statista数据），说明领域大趋势。\n"
            "   - 痛点层：将宏观问题收敛至具体工程/研究场景，用具体数据量化痛点（例如：'现有方法处理X任务时，响应延迟高达Y ms，导致Z%的业务损失'）。\n"
            "   - 切入点：明确指出本作品针对上述哪一具体缺口提出的解决方案。\n"
            "2. 禁止项：严禁使用'随着科技发展'、'人们生活水平提高'等空泛开头。必须引用证据材料中的具体数值或案例。\n"
            "3. 证据锚定：每一句结论性描述后建议标注证据来源索引（如 `[证据片段ID]`）。"
        ),
        "review": (
            "【问题背景槽位 - 严格复查规则】\n"
            "1. 逻辑链条检查：宏观趋势是否必然导致所述痛点？方案切入点是否覆盖了痛点的核心矛盾？"
            "2. 事实核查：检查所有数字、百分比、公司名称是否与提供的证据材料一致，避免AI幻觉造成的数值偏差。"
            "3. 篇幅控制：背景篇幅应控制在全文15%以内，避免头重脚轻。若发现长篇大论未切入主题，需提示精简。"
        ),
    },
    SLOT_TYPE_RELATED: {
        "generation": (
            "【相关工作槽位 - 高精度生成规则】"
            "1. 分类对比矩阵构建：将现有方案分为2-3个技术流派（如：基于规则的方法、基于传统机器学习、基于深度学习端到端）。"
            "2. 结构化分析模板（每个流派需包含）："
            "   - 代表工作：[方法名/论文] (年份) 核心机制简述。"
            "   - 优势：客观陈述其在该领域确立的基准贡献。"
            "   - 在本场景的局限性：结合本项目具体应用环境（如低算力、小样本、实时性要求），指出其不适应之处。"
            "   - 本方案改进思路：一句话预告本方案如何针对上述局限进行改良。"
            "3. 语气约束：使用'尚未完全解决'、'在处理X条件时性能退化'等客观学术表达，禁止使用'毫无价值'、'完全错误'等主观贬低词汇。"
        ),
        "review": (
            "【相关工作槽位 - 严格复查规则】\n"
            "1. 对比公平性：检查是否遗漏了关键的SOTA方法，或是否故意选择了过时的方法进行'稻草人对比'。\n"
            "2. 归因准确性：局限性分析是否基于真实测试条件？若无证据表明该方法在某条件下失效，应提示'缺乏实验证据支撑此局限判断'。\n"
            "3. 过度承诺检查：改进思路描述是否夸大了本方案的能力（例如本方案尚未验证大模型效果却声称解决了大模型幻觉问题）。"
        ),
    },
    SLOT_TYPE_TECHNICAL: {
        "generation": (
            "【技术方案槽位 - 高精度生成规则】"
            "1. 架构总览先行：必须先用文字或简图描述'输入-处理核心-输出'的数据流向，说明模块间的耦合关系（松耦合/紧耦合）。\n"
            "2. 技术选型博弈分析：对于每一个选用的核心算法/框架，必须附带一句 **'为什么不是...'** 的解释。"
            "   - 示例：'采用A算法而非B算法，是因为A算法在证据[E03]中展示出对小目标检测的召回率高出12%。'"
            "3. 关键技术深度剖析："
            "   - 数学表达：若涉及模型，提供核心公式或损失函数定义。\n"
            "   - 流程细节：描述数据在模块内部的具体变换过程，避免'首先处理数据，然后输入模型'这类流水账描述。"
        ),
        "review": (
            "【技术方案槽位 - 严格复查规则】"
            "1. 逻辑闭环检查：数据流转是否有缺失环节？输入特征维度与模型要求是否一致？"
            "2. 术语一致性：该槽位定义的术语（如'特征向量V'）是否与后续实现、测试章节保持命名统一？"
            "3. 可行性评估：方案描述是否包含了未经验证的、远超现有证据能力范围的技术黑箱（如'此处使用自研超强算力集群解决'但无硬件描述）。"
        ),
    },
    SLOT_TYPE_IMPLEMENTATION: {
        "generation": (
            "【系统实现槽位 - 高精度生成规则】"
            "1. **环境指纹信息：必须列出开发环境的'指纹级'信息，以证明真实落地。包括："
            "   - 硬件平台：CPU型号、RAM大小、GPU型号及显存。"
            "   - 软件栈：OS版本、Python/PyTorch/TensorFlow精确版本号、关键依赖库版本。"
            "2. 工程难点攻坚记录：挑选1-2个在编码调试中遇到的具体技术难题（Bug/性能瓶颈），描述 '现象-排查过程-最终解决方案'。若无此类记录，标注[此处可补充具体Debug案例]。"
            "3. 核心代码逻辑映射：描述模块时，应指出对应的关键文件或类名（如 `core/feature_extractor.py::ExtractFBank`），增强可追溯性。"
        ),
        "review": (
            "【系统实现槽位 - 严格复查规则】\n"
            "1. 工作量评估：检查描述是否过于简略，以至于看起来像是'配置环境+运行脚本'即可完成的教程级任务。"
            "2. 版本冲突检查：软件版本号组合是否合理（如 PyTorch 1.2 与 CUDA 12.0 不兼容）。"
        ),
    },
    SLOT_TYPE_TESTING: {
        "generation": (
            "【测试分析槽位 - 高精度生成规则】"
            "1. 数据集元信息：必须包含训练/验证/测试集划分比例、样本总量、类别分布情况。"
            "2. 指标定义公式：对每一个评价指标（如F1-Score, mAP, MAE），必须给出具体计算公式，消除歧义。"
            "3. 零容忍幻觉策略：所有表格中的数值必须严格来源于证据材料。"
            "   - 若证据仅有趋势图无精确值 -> 填写'≈约X.XX (基于图表目测)'。"
            "   - 若完全无数据 -> 整行留空并标注'[待填充: 需补充实验X在数据集Y上的测试结果]'。"
            "4. 对比基线设置：明确列出对比方法的来源（复现自开源代码/引用论文报告数据/本机实测）。"
        ),
        "review": (
            "【测试分析槽位 - 严格复查规则】\n"
            "1. 公平性审计：检查对比实验是否在同一硬件环境、同一数据集划分下进行。"
            "2. 结论保守性：检查结论是否过度解读微小提升。"
        ),
    },
    SLOT_TYPE_INNOVATION: {
        "generation": (
            "【创新点槽位 - 高精度生成规则】"
            "1. 数量限制与分级：输出1-3个核心创新点。若有多个，需标明优先级（Primary / Secondary Innovation）。"
            "2. 三元组描述法（每个点必须包含）："
            "   - What (区别)：明确指出与 [具体某篇相关工作]相比，本方案的独特结构/机制是什么。"
            "   - How (实现)：简述该创新点在代码/数学层面的具体实现形态。"
            "   - Evidence (佐证)：引用测试分析槽位中的具体表格行/图号作为效果支撑（例如：'该创新带来精度提升见表2第4行'）。"
            "3. 规避泛泛而谈：严禁使用'提出了一种新颖的方法'、'显著提升了性能'这类无实质内容的空洞表述。"
        ),
        "review": (
            "【创新点槽位 - 严格复查规则】"
            "1. **可证伪性检查**：创新点描述是否足够具体到可以被第三方复现或挑战？若描述模糊（如'优化了网络结构'但未说明如何优化），标记为不合格。\n"
            "2. **贡献归属**：检查是否将属于开源库/基础模型的功劳归功于本项目的创新（例如：'本项目的创新是使用了Transformer架构'——除非是特定修改的Transformer变体）。"
        ),
    },
    SLOT_TYPE_GENERAL: {
        "generation": (
            "【通用技术槽位 - 高精度生成规则】\n"
            "你正在处理一个未被具体分类的项目报告内容。请遵循以下通用原则：\n"
            "1. 术语定义前置：首次出现的专业缩略词必须给出全称及解释。\n"
            "2. 证据高亮：尽量引用证据材料中的具体文件名、日志片段或截图描述来佐证文字。\n"
            "3. 段落结构：遵循'Topic Sentence - Supporting Details - Concluding/Transition Sentence'结构。"
        ),
        "review": (
            "【通用技术槽位 - 严格复查规则】\n"
            "1. 术语漂移检查：同一概念是否在前后文中被交替使用了不同词汇（如时而叫'节点'，时而叫'服务器'）？\n"
            "2. 冗余检查：内容是否与前文已有的背景或技术方案章节发生机械重复？若是，建议合并或删除。"
        ),
    },
}


def infer_slot_type(slot: Dict[str, Any]) -> str:
    """根据slot_id/title/description进行轻量分类。"""
    text = f"{slot.get('slot_id', '')} {slot.get('title', '')} {slot.get('description', '')}".lower()

    patterns = [
        (SLOT_TYPE_REFERENCE, r"参考文献|reference|bibliograph|citation"),
        (SLOT_TYPE_TESTING, r"测试|实验|评估|指标|evaluation|experiment|benchmark|ablation|accuracy|precision|recall|f1|mape|mae"),
        (SLOT_TYPE_INNOVATION, r"创新|特色|亮点|novel|innovation|contribution"),
        (SLOT_TYPE_IMPLEMENTATION, r"系统实现|工程实现|部署|implementation|engineering|deployment|pipeline"),
        (SLOT_TYPE_TECHNICAL, r"技术方案|总体设计|架构|method|pipeline|framework|architecture"),
        (SLOT_TYPE_RELATED, r"相关工作|现有方案|对比|related work|baseline|sota"),
        (SLOT_TYPE_BACKGROUND, r"背景|来源|问题|痛点|需求|background|motivation|challenge|overview"),
    ]

    for slot_type, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return slot_type

    return SLOT_TYPE_GENERAL


def get_generation_skill(slot: Dict[str, Any]) -> str:
    slot_type = infer_slot_type(slot)
    base = _SKILL_RULES.get(slot_type, _SKILL_RULES[SLOT_TYPE_GENERAL])["generation"]
    if slot_type != SLOT_TYPE_REFERENCE:
        base = (
            f"{base}\n"
            "4. 引用格式约束：仅允许在段落内部使用文内引用（如 (Author, Year) 或 [1]），"
            "禁止在本槽位输出独立的参考文献列表。"
        )
    return base


def get_review_skill(slot: Dict[str, Any]) -> str:
    slot_type = infer_slot_type(slot)
    return _SKILL_RULES.get(slot_type, _SKILL_RULES[SLOT_TYPE_GENERAL])["review"]


def get_skill_snapshot(slot: Dict[str, Any]) -> Dict[str, str]:
    slot_type = infer_slot_type(slot)
    cfg = _SKILL_RULES.get(slot_type, _SKILL_RULES[SLOT_TYPE_GENERAL])
    return {
        "slot_type": slot_type,
        "generation_skill": cfg["generation"],
        "review_skill": cfg["review"],
    }

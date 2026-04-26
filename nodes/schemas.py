"""Pydantic 结构化输出模型，用于替换原有的 JSON 字符串解析。"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SlotDraft(BaseModel):
    """槽位生成草稿的结构化输出。"""

    slot_id: str = Field(..., description="槽位唯一标识")
    draft_text: str = Field(..., min_length=10, description="生成的段落文本，150-400字")
    source_refs: List[str] = Field(default_factory=list, description="证据来源编号列表")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")
    risk_notes: List[str] = Field(default_factory=list, description="潜在风险或不足说明")


class SlotReview(BaseModel):
    """槽位复查的结构化输出。"""

    slot_id: str = Field(..., description="槽位唯一标识")
    review_text: str = Field(..., description="2-4句具体问题说明")
    revised_text: str = Field(default="", description="建议性修订文本（可选）")
    source_refs: List[str] = Field(default_factory=list, description="复查后确认的证据来源")
    confidence: float = Field(..., ge=0.0, le=1.0, description="复查置信度")
    disagreements: List[str] = Field(default_factory=list, description="明确分歧/风险点列表")


class SlotRefine(BaseModel):
    """槽位改写的结构化输出。"""

    slot_id: str = Field(..., description="槽位唯一标识")
    draft_text: str = Field(..., min_length=10, description="改写后的段落文本")
    source_refs: List[str] = Field(default_factory=list, description="证据来源编号列表")
    confidence: float = Field(..., ge=0.0, le=1.0, description="改写后置信度")
    risk_notes: List[str] = Field(default_factory=list, description="剩余风险说明")


class OutlinePlan(BaseModel):
    """大纲规划的结构化输出。"""

    slot_order: List[str] = Field(default_factory=list, description="建议的槽位生成顺序")
    terminology_glossary: dict = Field(default_factory=dict, description="术语统一表")
    cross_slot_refs: dict = Field(default_factory=dict, description="跨槽位引用关系")
    data_reference_rules: List[str] = Field(default_factory=list, description="数据引用一致性规则")
    key_messages: List[str] = Field(default_factory=list, description="全文核心信息点")


class ConsistencyIssue(BaseModel):
    """一致性审查发现的单项问题。"""

    slot_ids: List[str] = Field(..., description="涉及槽位ID列表")
    issue_type: str = Field(..., description="问题类型：terminology/timeline/data/contradiction")
    description: str = Field(..., description="问题描述")
    suggestion: str = Field(default="", description="修改建议")


class ConsistencyReport(BaseModel):
    """一致性审查报告。"""

    issues: List[ConsistencyIssue] = Field(default_factory=list, description="发现的问题列表")
    overall_score: float = Field(default=1.0, ge=0.0, le=1.0, description="整体一致性评分")
    summary: str = Field(default="", description="审查总结")

"""OpenAPI descriptions only; raw pipeline results are returned without filtering."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Extensible(BaseModel):
    model_config = ConfigDict(extra="allow")


class TaskStatus(BaseModel):
    task_id: str
    filename: str
    language: str = Field(description="实际处理语言 en/zh；自动检测完成前为 auto。")
    requested_language: str = Field(description="上传时指定的 auto/en/zh，检测后保持不变，用于幂等匹配。")
    language_source: str = Field(description="pending / explicit / pdf_text / parsed_text / fallback_en。")
    status: str = Field(description="PENDING / RUNNING / DONE / FAILED / DEFERRED / PAUSING / PAUSED / CANCELLING / CANCELLED / DELETING。PAUSED 可恢复且 finished=false，CANCELLED 为终态。")
    stage: str = Field(description="当前阶段 parsing / mining / extracting；尚未开始或已结束时可为空字符串。")
    retry_pending: bool
    finished: bool
    attempts: int = Field(description="本轮启动次数，含自动重试；手工 resume 后重置为 0。")
    chunks_total: int | None = Field(description="片段总数，尚未确定时为 null。")
    chunks_done: int = Field(description="已完成片段数；不表示全流程百分比。")
    error: str | None
    created_at: float = Field(description="Unix 时间戳，秒。")
    updated_at: float = Field(description="Unix 时间戳，秒。")
    elapsed_seconds: float | None
    result_url: str = Field(description="可直接 GET 的完整结果 URL。")


class SourceBlock(Extensible):
    block_idx: int | None = None
    page_idx: int | None = Field(None, description="0-based 页索引。")
    bbox: list[float] | None = Field(None, description="[x0,y0,x1,y1]，原解析器坐标；MinerU 通常为 0–1000 的页面归一化坐标。")
    type: str | None = None


class SourceImage(SourceBlock):
    url: str = Field(description="已有图片的完整下载 URL。关联到 chunk，不保证是该反应的专属反应图。")


class ReactionSource(BaseModel):
    reaction_id: str
    chunk_id: str
    content_list_url: str | None
    position_scope: Literal["reaction", "chunk", "unavailable"]
    blocks: list[SourceBlock] = Field(description="反应原文坐标；若回退到整个片段，position_scope=chunk；无法定位时为空。")
    images: list[SourceImage] = Field(description="所属片段关联的已有图片；无图片时为空，不代表识别失败。")


class Downloads(BaseModel):
    reactions: str | None
    routes: str | None
    pdf: str | None = Field(description="交付归档中的源 PDF；导出未产生此文件时为 null。")
    chunks: str | None
    content_list: str | None
    document: str | None
    summary: str | None
    records: str | None = Field(description="原始追加历史 results.jsonl，可能含旧版本；不要直接作为当前业务结果入库。")
    images: str | None = Field(description="图像识别结果 JSON；图片本身的 URL 见 reaction_sources[].images[].url。")
    final_product: str | None
    delivery_manifest: str | None


class Coverage(Extensible):
    identified_examples: int | None = None
    covered_examples: int | None = None
    processed_examples: int | None = None
    products_with_structure: int | None = None
    products_total: int | None = None


class FinalProduct(Extensible):
    """文档级目标候选，用于展示；不替代 routes[].target。其余字段不稳定。"""

    product: str | None = None
    reference: str | None = None
    position: dict[str, Any] | None = None
    error: str | None = None


class DocumentResult(Extensible):
    """列出的状态、计数和覆盖字段可用于展示；其余内部字段不稳定。"""

    status: str | None = Field(None, description="procedure_processed / procedure_unverified / no_reaction_found / extraction_failed / partial。不是队列任务状态。")
    reason: str | None = None
    procedure_chunks: int | None = None
    successful_chunks: int | None = None
    needs_review_chunks: int | None = None
    non_reaction_chunks: int | None = None
    image_failures: int | None = None
    coverage: Coverage | None = None
    final_product: FinalProduct | None = None
    source_models: dict[str, Any] | None = None
    result_file: str | None = Field(None, description="内部文件系统路径；下载使用 downloads.records。")


class DeliveryStatistics(Extensible):
    reactions: int | None = None
    routes: int | None = None
    chunks: int | None = None
    missing_chunks: int | None = None
    source_reactions: int | None = None
    target_groups: int | None = None
    trusted_steps: int | None = None
    trusted_routes: int | None = None
    filtered_reactions: int | None = None


class DeliveryState(Extensible):
    status: Literal["ready", "error"]
    error: str | None = Field(None, description="status=error 时的导出错误。")
    statistics: DeliveryStatistics | None = Field(None, description="status=ready 时的导出统计。")
    directory: str | None = Field(None, description="内部服务器路径，不是 URL。")


class SummaryResult(Extensible):
    """delivery、result_status、图像失败计数和 coverage 可供开发使用；其余内部字段不稳定。"""

    delivery: DeliveryState | None = Field(None, description="缺键/null 表示无导出状态，不能按 ready 处理。")
    result_status: dict[str, int] | None = None
    image_failures: int | None = None
    failed_images: list[Any] | None = Field(None, description="内部失败详情，元素结构不稳定。")
    coverage: Coverage | None = None
    extraction_outcome: str | None = None
    elapsed_seconds: float | None = None
    final_product: FinalProduct | None = None
    source_models: dict[str, Any] | None = None
    source: str | None = None
    source_relative: str | None = None
    total_chunks: int | None = None
    by_kind: dict[str, int] | None = None
    segmentation_version: str | None = None
    coverage_gaps: list[Any] | None = None
    without_keyword_evidence: int | None = None
    empty_heading_ids: list[str] | None = None
    empty_heading_count: int | None = None
    heading_count: int | None = None
    covered_headings: int | None = None
    with_procedure: int | None = None
    cross_page: int | None = None
    position_matched: int | None = None
    text_reference_links: int | None = None
    global_dict_entries: int | None = None
    needs_review_chunks: int | None = None
    non_reaction_chunks: int | None = None
    extracted_chunk_ids: list[str] | None = None
    reclassified_chunk_ids: list[str] | None = None
    links_file: str | None = None
    chunks_file: str | None = None
    result_file: str | None = None
    final_product_file: str | None = None
    document_result_file: str | None = None


class FileInfo(Extensible):
    lines: int | None = None
    bytes: int
    sha256: str


class DeliveryManifest(Extensible):
    """processing_status、missing_chunks、statistics 可用于判断交付范围；归档细节不稳定。"""

    document_id: str
    processing_status: Literal["complete", "partial", "failed"]
    missing_chunks: list[str]
    processing_failures: list[dict[str, Any]] = Field(description="处理失败详情，元素结构按失败类型变化。")
    statistics: DeliveryStatistics
    processing_note: str | None = None
    schema_versions: dict[str, str] | None = None
    business_files: dict[str, FileInfo] | None = None
    archive_files: dict[str, FileInfo] | None = None
    archive_lookup: dict[str, str] | None = None


class InternalReaction(Extensible):
    """不稳定的单步内部结果；勿依赖其结构入库，业务反应用 result.reactions。"""

    index: int | None = None
    step_id: str | None = None
    status: str | None = None
    reason: str | None = None
    text: str | None = None
    step_source: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = Field(None, description="M1 抽取：ner、action、yield_info、stereochemistry 等，内部结构不稳定。")
    coref: dict[str, Any] | None = Field(None, description="指代消解后的抽取数据，内部结构不稳定。")
    validation: dict[str, Any] | None = Field(None, description="质量及 rxn_validation（表达式、映射、标准化化合物等），内部结构不稳定。")
    quality: dict[str, Any] | None = None


class InternalRecord(Extensible):
    """按当前 chunk 指纹筛选的最新内部结果，不稳定；不作为业务入库依据。"""

    id: str
    source: dict[str, Any] | None = None
    status: str | None = None
    reason: str | None = None
    error: str | None = None
    reactions: list[InternalReaction] | None = None
    position: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    coref: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    image: dict[str, Any] | None = None
    chunk_fingerprint: str | None = None
    elapsed_seconds: float | None = None
    context_version: str | None = None
    terminal_product_coverage: str | None = None


class InternalChunk(Extensible):
    """不稳定的分段结果；定位和图片优先使用 reaction_sources。"""

    id: str
    source: str | None = Field(None, description="content_list 的内部路径，不是 URL。")
    title: str | None = None
    text: str | None = None
    kind: str | None = None
    process_kind: str | None = None
    block_indices: list[int] | None = None
    page_start: int | None = Field(None, description="1-based 页码。")
    page_end: int | None = None
    images: list[str] | None = None
    source_parts: list[dict[str, Any]] | None = None
    steps: list[dict[str, Any]] | None = None
    position: dict[str, Any] | None = None
    reference_id: str | None = None
    number: int | str | None = None
    heading_type: str | None = None
    parent_example_id: str | None = None
    title_source: dict[str, Any] | None = None
    block_start: int | None = None
    block_end: int | None = None
    tables: list[Any] | None = None
    procedure_raw: str | None = None
    text_parts: list[Any] | None = None
    procedure: str | None = None
    references: list[Any] | None = None
    has_procedure: bool | None = None
    candidate_evidence: bool | None = None
    segmentation_version: str | None = None
    chunk_fingerprint: str | None = None
    procedure_index: dict[str, Any] | None = None
    context_version: str | None = None


class InternalImages(Extensible):
    """不稳定的图像识别中间结果，不依赖其结构入库。"""

    dict_result: dict[str, Any] | None = Field(None, description="图中化合物编号到结构的映射。")
    image_result: list[dict[str, Any]] | None = Field(None, description="识别到的图像反应结构，元素结构不稳定。")
    per_image_detections: list[dict[str, Any]] | None = None
    failed_images: list[Any] | None = None
    image_failures: int | None = None
    image_pipeline_version: str | None = None
    parse_fingerprint: str | None = None


class InternalParse(Extensible):
    """不稳定的 PDF 解析信息；服务器路径不供客户端拼接，使用 downloads。"""

    status: str | None = None
    content_list_file: str | None = None
    image_base: str | None = None
    parse_dir: str | None = None
    md_file: str | None = None
    images: int | None = None
    elapsed_seconds: float | None = None


class InternalManifest(Extensible):
    """不稳定的分段统计与来源信息。"""

    source: str | None = None
    source_relative: str | None = None
    total_chunks: int | None = None
    by_kind: dict[str, int] | None = None
    heading_count: int | None = None
    covered_headings: int | None = None
    coverage_gaps: list[Any] | None = None


class ResultData(BaseModel):
    reactions: list[dict[str, Any]] = Field(description="稳定业务反应，与 reactions.jsonl 相同；完整字段见交付数据字段说明。")
    routes: list[dict[str, Any]] = Field(description="稳定业务路线，与 routes.jsonl 相同；steps 按文档顺序，目标取 target。")
    downloads: Downloads = Field(description="已有文件的完整 URL；不存在、越出任务目录的文件为 null。")
    reaction_sources: list[ReactionSource]
    document: DocumentResult | None
    summary: SummaryResult | None
    delivery_manifest: DeliveryManifest | None
    final_product: FinalProduct | None
    warnings: list[str] = Field(description="读取产物遇到的问题，不等同于化学质量标记；缺失文件可能没有 warning。")
    records: list[InternalRecord]
    chunks: list[InternalChunk]
    images: InternalImages | None
    parse: InternalParse | None
    manifest: InternalManifest | None
    links: list[dict[str, Any]] = Field(description="不稳定的过程文字引用候选，不等同于路线物料连接；入库用 routes[].connections。")
    files_url: str = Field(description="兼容字段：文件接口前缀，不是可枚举目录；请使用 downloads 中的完整 URL。")


class TaskResult(TaskStatus):
    result: ResultData


RESULT_RESPONSES = {
    200: {"model": TaskResult, "description": "任务已结束；status 可能为 DONE 或 FAILED，可含部分结果。"},
    202: {"model": TaskStatus, "description": "排队、运行或自动重试中；继续查询 result_url。"},
}

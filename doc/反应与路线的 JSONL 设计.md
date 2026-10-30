# 按 PDF 交付反应与路线的 JSONL 设计

本文定义业务交付接口，面向后续几万个 PDF 的批量入库。`20261006_route_delivery` 是保留内部字段路径的旧版精简投影；本设计进一步将内部结果转换为统一的业务实体。`reaction-v1` 与 `route-v1` 已在导出器中实现，十篇样本的交付与验收说明见 [business-delivery-20261007.md](business-delivery-20261007.md)。以下“当前样本的设计依据”保留旧版统计；新版按本契约过滤无业务内容且无路线引用的失败占位。

## 1. 交付边界与设计原则

每个 PDF 只交付两个业务文件：

| 文件 | 每行的实体 | 主键与关联 |
|---|---|---|
| `reactions.jsonl` | 一个反应或制备步骤的候选记录 | `reaction_id` |
| `routes.jsonl` | 通向一个具体目标的一种制备方法 | `route_id`；通过 `steps` 引用反应 |

中间结果、模型输出、详细验证、错误日志和交付控制信息统一放在 `artifacts/` 中。两个 JSONL 不混入文档统计、错误日志或 meta 行。

1. **接口面向业务实体。** 系统不需要理解 `chunk → reaction → validation/ner`，也不需要自行选择 extraction、coref、validation 中的有效层。
2. **一个方法是一条路线。** 同目标的多个方法分别占一行，由 `target_group_id` 关联；路线内不再嵌套 `methods[]`，不再另设 `method_id`。
3. **关系只有一个权威来源。** 路线引用反应，不复制反应正文和化合物；反应不冗余输出 `route_ids`、`method_ids`。系统从 `steps` 建立关联表。
4. **保留事实和必要的不确定性。** 业务层保留来源、收率性质、立构语义和少量质量结论，详细判断过程留在归档。不能为了精简把未知值变成确定事实。
5. **来源身份与化学身份分开。** 主键依据文档及来源位置生成；结构、名称、晶型等用于检索和目标分组。
6. **按文档发布完整快照。** 同一快照内两个文件的引用必须闭合；重跑时整体更新，避免旧步骤或旧路线残留。
7. **稳定排序、固定类型。** 按原文出现顺序输出，布尔值使用 JSON boolean，字段含义和类型由交付契约定义。

### 当前样本的设计依据

对 `20261006_route_delivery` 十篇结果的核对显示：

- 共 343 个 chunk、779 条反应候选、218 个目标分组、283 个方法。原有目标分组不能直接等同于一条制备路线；202110246110.6 一个目标下有 55 个方法。
- 773 条候选有标准化化合物层，6 条需要使用后备业务层。
- 149 条收率为计算值；反应级立构字段包含相对构型和混合物信息，不能仅保留化合物自身的立构字段。
- 步骤原文合计约 475 KB，约占现有精简反应 JSONL 体积的 12%，值得保留用于展示和核查。
- 当前严格规则下只有 3 条反应符合自动采用门槛，283 个方法均未符合。候选记录入库与作为已验证数据使用必须区分。

以上是现有样本统计，不代表新接口已经导出，也不代表抽取准确率或全文覆盖率。

## 2. 主键、版本与重跑

### 文档身份

`document_id` 使用原 PDF 文件名（包含 `.pdf` 后缀），例如 `US10040872.pdf`。同一输入文件夹内文件名已经唯一；同一批次汇集多个文件夹且遇到同名文件时，使用 `结果文件夹名/原文件名`，例如 `batch_a/US10040872.pdf`。不使用 SHA-256 作为文档主键或结果目录索引。

已选定或已交付的文件名身份在后续追加任务时保留；新增同名任务加结果文件夹前缀。PDF 改名会改变文件名身份，需要系统侧显式处理更名。同名 PDF 内容更新使用同一文档身份和新的交付版本。PDF 的现有校验哈希仅用于完整性和原件回查。

### 反应与路线身份

- 第一版可以使用 `reaction_id = document_id + "::" + chunk_id + "::" + reaction_index`。它保证同一分块结果中的身份唯一，但重新分块、拆分步骤或调整序号后可能变化。
- 后续若具备稳定的原文步骤定位，可改为依据来源锚点生成反应 ID。不能声称仅靠 `chunk_id + index` 就能跨所有重跑保持稳定。
- 路线 ID 使用可读来源编号：`reaction_id + "::route::" + product_number`；仅标题目标使用 `title`。产物序号从 1 开始，源于终点步骤的产品列表。同一终点的多个目标分别编号，不使用哈希或 SMILES 作路线主键。
- `compound_id` 在一条反应内唯一，例如 `c1`、`c2`。完整化合物引用为 `(reaction_id, compound_id)`，不是跨文档的化合物身份。
- `target_group_id` 取组内第一种来源方法的终点：`reaction_id + "::target::" + product_number`，是文档内的可读候选分组编号。成员关系随快照更新，不用于证明跨 PDF 的化学等价性。

ID 中的分隔符 `:` 和转义符 `%` 做必要转义，中文、空格等保留可读性。反应、路线、目标分组均不使用哈希 ID；现有文件校验和来源指纹仍留在归档。

### 格式版本与快照版本

- 每条业务记录携带 `schema_version`，分别为 `reaction-v1`、`route-v1`。
- `quality.policy_version` 表示质量判断规则版本，与 schema 版本分开。
- manifest 携带 `delivery_revision`、导出器版本和两个文件的哈希。`delivery_revision` 标识该 PDF 的一次完整交付，不是逐条业务字段。
- 相同输入及相同配置的重复导出应具有相同记录顺序、ID 和业务内容。新增可选字段可以兼容扩展；字段语义或实体粒度变化需要升级 schema。

## 3. reactions.jsonl：一行一个反应或制备步骤

以下为格式示意，短 ID、名称及条件用于说明接口，不是样本中的实际化学结果。展示时采用缩进；实际 JSONL 每个对象占一个物理行，正文换行编码为 `\n`。

```json
{
  "schema_version": "reaction-v1",
  "document_id": "doc_001",
  "reaction_id": "rxn_001",
  "process_kind": "synthesis",
  "compounds": [
    {
      "compound_id": "c1",
      "roles": ["reactant"],
      "name": "原料 X",
      "canonical_smiles": null
    },
    {
      "compound_id": "c2",
      "roles": ["product"],
      "name": "中间体 A",
      "canonical_smiles": null
    }
  ],
  "conditions": {
    "temperature": "80℃",
    "time": "2小时",
    "yield": {
      "kind": "explicit",
      "value": 85,
      "unit": "%"
    }
  },
  "actions": ["加入", "加热", "搅拌"],
  "reaction_smiles": null,
  "stereochemistry": {
    "kind": "unspecified",
    "descriptors": [],
    "structure_coarsened": false
  },
  "source": {
    "chunk_id": "example_1",
    "reaction_index": 1,
    "pages": [10],
    "text": "该步骤的原文"
  },
  "quality": {
    "admission_eligible": false,
    "blocking_issues": ["product_structure_unavailable"],
    "policy_version": "2026-10-04-v9.1"
  }
}
```

### 字段契约

| 字段 | 类型 | 含义与约束 |
|---|---|---|
| `schema_version` | string | 反应接口版本 |
| `document_id`、`reaction_id` | string | 文档 ID、全局唯一的步骤 ID |
| `process_kind` | string | 第一版沿用可确认的 `synthesis`、`formulation`、`purification`；无法确定为 `unknown`。现有 `unverified` 转为 `unknown`，不新增无依据的分类 |
| `compounds` | array | 一份统一化合物列表；缺结构时仍保留有依据的名称和角色 |
| `conditions.temperature`、`conditions.time` | string 或 null | 当前抽取的条件文本，保留范围、回流、室温等表达；不擅自补单位或转换为单个数值 |
| `conditions.yield` | object 或 null | 有语义依据的收率信息；无法判断其报道或抽取状态时为 null |
| `actions` | string[] | 已抽取的动作序列；不代表已结构化所有操作参数 |
| `reaction_smiles` | string 或 null | 最终有效反应表达式；不承诺包含原子映射 |
| `stereochemistry` | object 或 null | 反应级产物立构语义；`unspecified` 与无法抽取的 null 不等价 |
| `source` | object | 步骤原文、定位及必要的程序来源引用 |
| `quality` | object | 当前质量规则计算的最终采用资格和阻断原因 |

化合物必备字段为 `compound_id`、`roles`、`name`、`canonical_smiles`。`name` 可以为 null，但不能在缺名称及结构时制造化合物占位。存在实际内容时，可增加 `cxsmiles` 和化合物自身的 `stereochemistry`。

`roles` 使用 `reactant`、`product`、`reagent`、`solvent`、`participant`。同一物料因映射补丁兼具试剂和反应参与物角色时，可以使用多个角色；仅在来源证据确认是同一物料记录时合并，不能仅因名称或 SMILES 相同就合并不同投料、不同批次或不同用途。

### 统一数据来源，而不是只保留某个内部层

1. 化合物优先读取 `validation.rxn_validation.normalized_compounds`。该层明确给出的空列表不能被后备层自动覆盖。
2. 标准化层缺失时，按最后可用业务层读取 validation、coref、extraction 的化合物，再转换为同一份 `compounds`。系统侧不接收这些内部层级。
3. `canonical_smiles` 只接收有效、无已知结构冲突的标准化结构。后备结构需经过规范化和检查；不能将未知来源的 `smiles` 直接改名为权威结构。无法确认时输出 null，原始候选结构留在归档。
4. 缺少标准化层不意味着删除整条记录；保留可确认的名称、角色、条件和原文，质量结论保持需要核查。
5. `reaction_smiles` 使用已确定的 `effective_expression`。旧记录缺少该字段时，只有 `patch_accepted=true` 才可选择已接受的 `best_expression`，否则使用可确认的 `base_expression`；没有有效表达式则为 null。
6. 不能把所有溶剂、试剂机械拼入反应表达式。内部 `ignored` 标记不代表业务物料不存在，不得因此删除实际使用的溶剂或试剂。

### 收率、立构与用量

收率保留以下语义：

| `yield.kind` | 业务含义 | `value` |
|---|---|---|
| `explicit` | 原文报道且当前抽取值与报道匹配 | 数值 |
| `derived` | 依据原文数量计算 | 数值，附 `basis` |
| `not_reported` | 在已检查的来源中未发现报道 | null |
| `not_extracted` | 来源中发现报道，但未可靠抽取 | null，保留 `reported_values` |
| `unattributed` | 抽取数值缺乏可靠归属 | null，候选值保留在归档 |

`value` 和可确认的 `reported_values` 使用 JSON number，百分收率的 `unit` 为 `%`。转换失败不能填零，应输出未知并记录问题。原始字符串与详细匹配证据保留在归档。`derived` 的 `basis` 保留实际计算依据，例如底物及产物的 mmol、公式和舍入位数；不能伪装为原文报道收率。

反应级 `stereochemistry` 保留 `kind`、`descriptors`、`structure_coarsened`。相对构型、混合物以及结构被简化的信息不能仅由普通 canonical SMILES 替代；有实际内容的 `cxsmiles` 也应保留。

当前没有可靠化合物用量抽取时，不输出恒空的 `amount`、`unit` 占位字段。后续可增加可选的 `quantities`，以数组表示质量、物质的量、体积、当量或浓度，并携带单位和依据；不能只用一个无单位的 `amount`。原文已有数量但尚未抽取时，不能标记为“原文未报道”。

### 来源与程序引用

- `source.text` 保留该步骤原文，不能用模型总结替代。必要的通用程序原文可通过来源引用回查，不重复拼入多个步骤正文。
- `source.pages` 统一为 PDF 页序，从 1 开始，不等同于印刷页码。优先由步骤自身定位生成，不能直接复制整个 chunk 的页范围。
- 当前 `page_idx` 与 `page_start` 编号基准不同，导出时必须显式转换。无法定位时使用空数组，不猜页码。
- 条件来自通用程序时，可增加 `source.references`，每项包含 `kind`、文档内 `reference_id`、`status` 和 `pages`。`kind` 区分 `procedure` 与其他来源；未解决的引用同时阻断自动采用。
- 方法参考不是物料依赖，不能据此创建路线连接。具体字段的继承和覆盖证据保留在归档，质量判断仍检查来源归属。
- `chunk_fingerprint`、完整坐标和机器绝对路径放在归档，回查时结合本次 manifest 确定对应原始记录。

### 哪些记录进入反应文件

保留真实反应候选，以及路线所需的制剂、结晶等制备步骤。`needs_review` 不是删除理由，制剂也不应因映射器判定 `non_reaction` 就直接删除。

纯检测、分析段落以及没有有效业务内容、也没有路线引用的失败占位放入归档。若路线需要引用抽取失败的步骤，保留带真实来源的候选节点，并明确质量阻断，不补造原料、产物或连接。

未归入任何路线的有效反应仍保留。系统从关联表反查即可识别；具体未归属原因放在 `artifacts/document_result.json` 等归档记录中，不另建业务清单。

## 4. routes.jsonl：一行一种制备方法

以下示例假设反应文件同时包含 `rxn_001` 和 `rxn_002`。各化合物 ID 在各自反应内解释。

```json
{
  "schema_version": "route-v1",
  "document_id": "doc_001",
  "route_id": "route_001",
  "target_group_id": "target_001",
  "target_group_basis": "exact_name",
  "title": "目标 B 的制备方法一",
  "process_kind": "synthesis",
  "target": {
    "reaction_id": "rxn_002",
    "compound_id": "c2"
  },
  "steps": ["rxn_001", "rxn_002"],
  "connections": [
    {
      "from": "rxn_001",
      "to": "rxn_002",
      "product_id": "c2",
      "input_id": "c1",
      "basis": "exact_name"
    }
  ],
  "quality": {
    "admission_eligible": false,
    "blocking_issues": [
      {
        "code": "step_quality_blocked",
        "reaction_ids": ["rxn_001", "rxn_002"]
      }
    ],
    "policy_version": "2026-10-04-v9.1"
  }
}
```

### 路线与具体产物

- `route_id` 标识一种来源制备方法，不承诺其已验证或可完整执行。
- `target` 引用最终步骤中的具体产物，不重复整个产物对象。多产物步骤必须指明具体 `compound_id`，不能用最后一步代替目标身份。
- 目标只能从来源标题获得时，`target` 保留对应 `reaction_id`、`compound_id: null`、实际标题名称 `name` 和 `basis: "source_title"`；加入 `target_from_title_without_extracted_product` 阻断。不能为填满引用而制造产物。
- `target_group_basis` 明确分组依据，可为 `canonical_smiles`、`exact_name`、`source_formula_label`、`parent_name` 或 `source_title`。只有标题依据的分组是候选分组，不代表已确定结构。
- 同一明确母体的晶型、盐型、溶剂合物可以归组，但每条路线的 `target` 仍指向具体形式。不同共价结构、明确不同立体异构体分开；配方不能仅按活性成分 SMILES 合并。式号只在当前文档内解释。

### 步骤顺序与依赖图

- `steps` 是去重后的步骤列表，按原文出现顺序排列，用于展示和建立成员关系。
- `connections` 是物料依赖图；相邻的 `steps` 不代表一定相连。分支、汇合、共享上游和一锅多步不能强行展开成一条线性链。
- 连接的 `product_id` 属于 `from` 反应，`input_id` 属于 `to` 反应。两者的物料角色和匹配依据必须可核查，不能仅凭反应 ID 表示所有物料都发生了关联。
- `basis` 保留实际匹配依据，如 `canonical_smiles`、`exact_name`、`material_reference`。存在明确物料来源引用时，可增加 `evidence_text` 保留引用句。
- 名称匹配或结构匹配表示规则识别的依赖，不自动证明实验批次相同；不能把方法参考当作已报道的物料来源。
- 多个上游候选放入 `quality.blocking_issues` 的 `candidate_reaction_ids`，不能任意选一个放进确定连接。候选可位于当前文档的其他方法中，不必强行加入本路线。
- 已知断开的来源步骤可保留在 `steps` 中并标明 `disconnected_procedure_steps`。无报道中间体的一锅反应不补造结构；沿用当前规则标记需要核查。

## 5. 质量结论与采用门槛

业务层只输出一份最终质量结论，不要求系统重复推导内部状态。

- 反应 `quality` 使用 `admission_eligible: boolean`、`blocking_issues: string[]`、`policy_version: string`。
- 路线 `quality` 使用同样的资格和版本字段；`blocking_issues` 为对象数组，包含 `code` 及必要的步骤、候选或引用 ID。
- 路线汇总不逐项复制所有步骤问题。使用 `step_quality_blocked` 指向不符合门槛的步骤，具体原因回查反应记录；路线自身的歧义、断点和目标缺口随路线保留。
- `mapping.confidence`、`diagnosis`、`patch_accepted`、`chunk_status`、结构状态、详细完整性判断、原子检查等参与内部评估，完整信息保留在归档。

导出器计算反应资格时沿用当前严格规则：质量证据存在且版本正确、原质量资格为真、无阻断、reaction 和 chunk 状态均为 `ok`、映射为 `passed` 或 `corrected`、不需要人工复核。缺证据、未知版本或字段缺失时默认不具备自动资格，不能因压缩接口而放宽门槛。

路线只有所有必要步骤合格、目标明确且无连接问题时才具备自动资格。当前基于合成验证的规则不为制剂自动放行；将来需要其他制备过程的采用规则时，应明确升级策略。

`admission_eligible=true` 的契约是“符合标明版本的自动采用门槛”，不是人工验证证明。系统仅对支持的策略版本自动采用；未知版本进入核查。

**候选保存与可信采用分开：** 格式和引用合法的候选可以入库用于展示、检索和人工核查；可信数据视图再根据 `admission_eligible` 和支持的策略版本过滤。不能把资格为假等同于业务内容不存在，也不能仅因映射置信度高就放入可信视图。

## 6. 空值与类型约定

| 值 | 统一含义 |
|---|---|
| `null` | 当前没有可确认的值；不单独表示原文未报道 |
| `[]` | 当前没有确认的成员；不证明原文不存在这些成员 |
| 字段缺省 | 仅适用于契约允许的可选扩展；不代表空值 |
| `""` | 业务字段不使用空串表达未知；旧数据中的空串在转换时规范化，原值留在归档 |
| `false` | 明确的布尔值，不能写成字符串 `"false"` |

本设计是业务 schema 转换，不是旧结果的路径、值、类型逐项透传。转换必须有明确规则：数值转换、页码转换、未知值规范化等可改变表示，不能改变事实或隐藏抽取失败。凡需要区分“未报道”“未抽取”“计算所得”的字段，使用明确的语义标签。

## 7. 逐篇目录、manifest 与原子交付

直接写到每篇解析结果文件夹内的 `delivery/`，不另建 SHA-256 目录，不生成批次总表、批次 manifest 或共享汇总索引。

```text
<解析后的每篇结果文件夹>/delivery/
  reactions.jsonl
  routes.jsonl
  artifacts/
    manifest.json
    results.jsonl
    chunks.jsonl
    document_result.json
    final_product.json
    ...模型输出、验证证据、日志、定位索引和其他中间产物
```

归档可保留原始目录结构，也可引用持久可访问的归档位置，不必复制所有大型缓存。不能仅记录接收方无法访问的生产机器绝对路径。manifest 与原始记录共同支持本次快照回查，包括 chunk 指纹和旧 ID 对新 ID 的映射。

manifest 至少包含：

- `document_id`、`pdf_sha256`、原文件名及已确认的专利编号。
- `delivery_revision`、导出器版本、两个业务 schema 版本、质量策略版本。
- `processing_status`：`complete`、`partial` 或 `failed`，表示抽取任务处理状态，不表示化学正确或全文召回完整。
- 已知缺失步骤、解析失败或未处理部分及其说明；统计只能描述已识别的处理范围，不能用已处理 chunk 比例证明全文覆盖。
- 两个 JSONL 的行数、字节数和 SHA-256，以及原始输入指纹和归档回查信息。

**空文件不能独立表示成功。** 只有收到完整交付控制信息、处理状态明确、文件哈希及零行计数匹配时，才可解释为空结果。处理完成且无记录表示本次未抽取到结果，不证明 PDF 中不存在任何反应。文件缺失、部分处理和失败必须分别表达。

manifest 是交付控制信息，不是第三份业务实体文件。若系统只传输两个 JSONL，任务接口或消息中必须携带等价的控制信息；不能依赖接收方读取未传输的 `artifacts/`。

导出时先在暂存目录生成并校验两个文件及 manifest，再发布整个文档快照。接收端按 `delivery_revision` 暂存核验后切换版本，不分别暴露不同快照的反应和路线。大型归档可独立存储，但必须保证快照可回查。

## 8. 入库与验收约定

### 文档快照更新

1. 接收系统以 `document_id` 识别文档，以 `delivery_revision` 识别交付版本；相同版本及哈希的重复提交应幂等。
2. 先校验 schema、类型、主键、引用和文件完整性，再建立候选数据及关系。控制信息与业务记录的文档身份必须一致。
3. 校验通过后，在事务或等价的版本切换机制中更新该文档的反应、化合物、路线和关联；移除本次快照不再包含的旧实体。
4. 失败或部分交付不得自动覆盖上一版完整数据。可以保存为待核查快照，并在文档处理状态中明确显示；完整零结果快照可以更新旧数据。
5. 人工核查结论另存，并记录其对应快照；重跑后若步骤身份或内容改变，不能将旧核查结论静默套用到新记录。

### 最低验收检查

- 每行是独立 JSON object，无注释、占位伪 SMILES、meta 行或 CSV 式字符串布尔值。
- 反应与路线主键唯一；每个反应中的化合物 ID 唯一，角色属于契约允许的集合。
- 每条路线的 `steps` 均可在同一 PDF 的反应快照中找到，且不重复。
- 确定连接的端点都属于本路线；化合物引用存在且角色正确；物料依赖图无环。歧义候选引用也必须可在当前文档中找到。
- 目标引用存在且指向产物；标题目标的空化合物引用必须有明确来源和阻断问题。
- `admission_eligible=true` 时无阻断，策略版本受支持；路线资格与全部必要步骤资格一致。
- 明确收率有报道依据，计算收率有计算依据，缺失或未归属值未被写成确定值；立构语义未被普通 SMILES 覆盖。
- 原文与步骤页码可回查，程序引用没有被转换成物料依赖，后备化合物和实际溶剂没有因精简而丢失。
- 两个文件的行数和哈希与控制信息一致，已知缺失或过滤记录在归档中可追踪。

系统侧可采用文档、反应、化合物出现记录、路线、路线步骤成员和物料连接等表承接这两个接口；目标展示按 `target_group_id` 汇总即可。当前实现按上述实体粒度和字段语义逐篇离线导出与验收，不需要为格式调整重新调用化学抽取模型。

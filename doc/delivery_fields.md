# ChemReX delivery 交付数据字段说明

- 适用范围：每篇文档目录下 `delivery/reactions.jsonl` 和 `delivery/routes.jsonl`。**业务入库只依赖这两个文件**。PDF 定位、图片展示和处理状态可使用 [API 响应中的辅助字段](ChemReX%20PDF%20API.md)，无需解析内部中间产物。

## 1. 通用约定


| 项             | 约定                                                                                                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 文件格式       | JSONL(UTF-8,无 BOM),一行一条独立 JSON 对象,无空行、无续行                                                                                                                                           |
| 目录与文档对应 | 一篇文档一个目录；API 目录由 `task_id` 标识，离线批次可能使用 `{专利号}_{源文件hash}`。`document_id` 通常为原 PDF 文件名；批次内同名文件可能使用 `{结果目录名}/{文件名}` 区分，不应自行拼接 |
| 主键           | `reactions.jsonl` → `reaction_id`;`routes.jsonl` → `route_id`;                                                                                                                                    |
| ID 格式        | `reaction_id` = `{document_id}::{chunk_id}::{reaction_index}`,如 `US10004750.pdf::synthesis_0::1`;`route_id` = `{目标reaction_id}::route::{序号}`,如 `US10004750.pdf::preparation_0_3::1::route::1` |
| 空值约定       | 可空字段用 JSON `null` 表示；部分嵌套字段可能缺键（见 §5.2）。嵌套对象为 null 时不再读取其子字段 |
| 枚举与扩展 | `process_kind` 当前集合为 `synthesis` / `formulation` / `purification` / `unknown`；化合物 roles 当前集合见 §2.2。问题代码、引用类型等属于可扩展字符串。客户端遇到未知值应保留原值并按“未知/其他”展示，不丢弃整条记录；未知附加字段可忽略或原样保存 |

## 2. reactions.jsonl(schema: reaction-v1)

一行一条**反应**记录。

### 2.1 顶层字段


| 字段              | 类型            | 必填 | 说明                                                                                                                                                     |
| ----------------- | --------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`  | str             | 是   | 恒为`reaction-v1`,可作版本标记存储                                                                                                                       |
| `reaction_id`     | str             | 是   | 全局唯一主键,格式见 §1                                                                                                                                  |
| `document_id`     | str             | 是   | 所属文档,如`US10004750.pdf`                                                                                                                              |
| `record_kind`     | str             | 是   | 记录类别，常见 `reaction_candidate`、`procedure_template`；可扩展，不替代 `process_kind` |
| `process_kind`    | str             | 是   | 当前取值集合：`synthesis` / `formulation` / `purification` / `unknown`；无法归类的上游值导出为 `unknown` |
| `compounds`       | array\<object\> | 是   | 参与化合物列表，可为空数组（如仅保留操作信息的待审记录），见 §2.2 |
| `conditions`      | object          | 是   | 反应条件(见 §2.3)                                                                                                                                       |
| `reaction_smiles` | str\| null      | 是   | 反应 SMILES，通常为 `左侧组分>>产物`；null 表示未能生成或存在结构冲突。组分规则见下文 |
| `actions`         | array\<str\>    | 是   | 操作动作序列,英文动词短语,**未归一化**(存在 `add`/`added`/`Add` 等形态与大小写变体),如 `dissolve`、`stir`、`filter`、`wash`;可为空数组(样本 82/222 为空) |
| `stereochemistry` | object\| null   | 是   | 立体化学信息；未抽取到时为 null（见 §2.4） |
| `quality`         | object          | 是   | 质检结果(见 §2.5)                                                                                                                                       |
| `source`          | object          | 是   | 溯源信息(见 §2.6)                                                                                                                                       |

`reaction_smiles` 的当前生成规则：左侧由可用于映射的 `reactant` 和 `participant` 构成，右侧为产物；试剂经判定参与反应并接受修正后可能作为 participant 加入左侧，溶剂不会自动全部加入。它不是完整的试剂、溶剂清单，完整参与物应从 `compounds.roles` 读取。通用表达式解析应支持 `左侧>中间组分>右侧`，不要依赖中间部分恒为空。

### 2.2 compounds[] 元素


| 字段               | 类型         | 必填 | 说明                                                                                                                                  |
| ------------------ | ------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `compound_id`      | str          | 是   | **仅在当前 reaction 内唯一**的局部编号(`c1`、`c2`…,每条反应从 c1 重新编号)。全局定位一个化合物需 `(reaction_id, compound_id)` 二元组 |
| `name`             | str\| null   | 是   | 提取的原文名称；仅有可解析结构时可为 null；与 `canonical_smiles` 不会同时为空 |
| `canonical_smiles` | str\| null   | 是   | 标准 SMILES;null 表示结构未解析出(样本中约 11% 为 null,如"seed crystal A"等非分子实体)                                                |
| `roles`            | array\<str\> | 是   | 角色，≥1 个；当前集合：`product` / `reactant` / `reagent` / `solvent` / `participant`。同一化合物可兼任多个角色 |

化合物还可能带 `cxsmiles`、`stereochemistry`、`resolution_status`、`resolution_trace`、`discrete_form_evidence` 等附加信息；这些键不保证存在，不作为基本入库字段的前提。

### 2.3 conditions


| 字段                               | 类型           | 必填   | 说明                                                                                                                                    |
| ---------------------------------- | -------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `conditions.temperature`           | str\| null     | 是     | 温度,**原文自由文本、未归一化**;多段操作以 `"; "` 分隔,如 `"room temperature; 4°C"`、`"dry ice-ethanol bath; -40°C; -20°C"`          |
| `conditions.time`                  | str\| null     | 是     | 时间,同为原文自由文本,可多段,如`"6 days; 16 hours; 90 minutes"`                                                                         |
| `conditions.yield`                 | object\| null  | 是     | 收率键恒存在；未抽取到有效收率信息或证据不足时为 null。下列子字段仅在对象非 null 时适用 |
| `conditions.yield.kind`            | str            | 是     | `explicit`（原文明确报告且可归属）/ `derived`（按原文物质量计算）/ `not_reported`（原文未报告）/ `not_extracted`（未能抽取或归属）/ `unattributed`（无法归属） |
| `conditions.yield.unit`            | str            | 是     | 恒为`%`                                                                                                                                 |
| `conditions.yield.value`           | float\| null   | 是     | `explicit` 的单值收率或 `derived` 的计算收率（如 `49.0`）；区间收率及其他 kind 为 null |
| `conditions.yield.reported_values` | array\<float\> | **否** | 可选键,仅`kind=not_extracted` 时出现,存放原文报告但未能归属的数值(如 `[83.0, 83.0]`)                                                    |
| `conditions.yield.range` | object | 否 | 原文报告区间时出现，含 `lower`、`upper` 两个数值；此时 `kind=explicit`、`value=null` |
| `conditions.yield.raw` | str | 否 | 区间或近似收率的原始表达 |
| `conditions.yield.approximate` | bool | 否 | 区间或近似收率附带的近似标记；区间不一定是近似值 |
| `conditions.yield.reported_ranges` | array\<object\> | 否 | `not_extracted` 时可能出现；元素含 `lower`、`upper` |
| `conditions.yield.basis` | object | 否 | `derived` 时必有；含 `substrate_mmol`、`product_mmol`、`decimals`（数值）及 `formula="100 * product_mmol / substrate_mmol"`（字符串），可带其他证据字段 |

例如 `{"kind":"explicit","unit":"%","value":null,"range":{"lower":70.0,"upper":80.0},"raw":"70–80%","approximate":false}`；不能把区间擅自转成单值。

### 2.4 stereochemistry（立体化学，入库可先不关注）

该字段可为 null。以下子字段仅在对象非 null 时适用。

| 字段                                  | 类型         | 必填 | 说明                                                     |
| ------------------------------------- | ------------ | ---- | -------------------------------------------------------- |
| `stereochemistry.kind`                | str\| null   | 是   | 常见 `unspecified` / `absolute`；允许其他描述类别；没有可用类别时为 null |
| `stereochemistry.descriptors`         | array\<str\> | 是   | 立体描述符,如`["(3R,5R)"]`;`kind=unspecified` 时为空数组 |
| `stereochemistry.structure_coarsened` | bool         | 是   | 结构是否被粗化(去立体/简化处理),样本中均为 false         |

### 2.5 quality（内部质量控制，用于后续改进提取质量，先可不关注，先全入库）


| 字段                         | 类型         | 必填 | 说明                                                                                                                                    |
| ---------------------------- | ------------ | ---- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `quality.admission_eligible` | bool         | 是   | 是否通过准入策略。false 的结果仍可入库，但可能缺少结构、收率或证据，需保留标记并按需复核 |
| `quality.blocking_issues`    | array\<str\> | 是   | 拦截原因**代码字符串**数组(注意:此处是字符串数组,与 routes.jsonl 中的对象数组结构不同,见 §5.2)。                                       |
| `quality.policy_version`     | str          | 是   | 质检策略版本,如`2026-10-07-auditfix`                                                                                                    |

### 2.6 source


| 字段                    | 类型            | 必填   | 说明                                                     |
| ----------------------- | --------------- | ------ | -------------------------------------------------------- |
| `source.chunk_id`       | str             | 是     | 来源分块标识,如`synthesis_0`、`example_4`、`step_2`      |
| `source.pages`          | array\<int\>    | 是     | 原文 PDF 页码,**1-based**;可为空数组(样本 15/222 为空)   |
| `source.reaction_index` | int             | 是     | 该 chunk 内的反应序号,从 1 开始                          |
| `source.text`           | str\| null      | 是     | 来源原文文本；缺失时为 null，并带相应质量问题 |
| `source.references`     | array\<object\> | **否** | 可选键:该反应引用了其他 procedure 时才存在。元素结构见下 |

**`source` 与 chunk 源文的链接关系**

API 调用方优先使用 `result.reaction_sources`，按 `reaction_id` 关联。API 已提供坐标、图片 URL 和 content_list URL；两个交付 JSONL 不增加这些字段。


| 开发需求                  | 用哪个字段              | 说明                                                         |
| ------------------------- | ----------------------- | ------------------------------------------------------------ |
| 展示反应的来源原文        | `source.text`           | 原文已内嵌                                                   |
| 定位 PDF 页码             | `source.pages`          | 反应实际所在页(1-based)                                      |
| 区分同一 chunk 的多条反应 | `source.reaction_index` | chunk 内序号,从 1 开始。多数 chunk 只产 1 条反应,少数产 2~3 条 |
| 定位到 PDF block 级坐标 | API `reaction_sources[].blocks` | `position_scope=reaction` 为反应级；`chunk` 表示仅能定位整个片段；`unavailable` 表示无坐标 |
| 展示原图 | API `reaction_sources[].images[].url` | 完整下载 URL；图片关联到所属 chunk，可能多步共用，不保证是单条反应的专属反应图 |

离线排查时也可用 `source.chunk_id` 查 `chunks.jsonl`，再按 `block_indices` 查 chunk.source 指向的 content_list 数组（不是 `block_start`/`block_end` 的连续包络）。中间文件结构不稳定；API 下载使用 `result.downloads` 返回的 URL，不根据服务器绝对路径拼接。

`source.references[]` 的用途:**跨段落的过程引用**。

专利常见写法是"按 Example 18 的方法制备,将 A 替换为 B"——本反应的实验步骤**不在本段原文里**,而是引用文中另一处 procedure。`source.references` 记录这种引用关系,告诉下游"这条反应的完整操作步骤在别的 chunk"。两种状态:

- **`status=resolved`**:引用已在本文档内定位,`target` 给出被引用的 chunk 及 block 级坐标。
  例:US10005783 的反应 `example_19::1`,原文 "Prepared by the method as described in Example 18, substituting 1-fluoro-... with 1-chloro-...",引用解析到 `target.chunk_id = "example_18"`——那个 chunk 里才是完整的操作步骤。
- **`status=missing`**:引用目标在本文档中找不到,引用悬空。
  例:US10004750 的反应 `example_10::1`,原文 "...obtained in the same manner as in Example 1",但该文档不存在 id 为 `example_1` 的 chunk。此类反应会同时带 blocking_issue `procedure_reference_missing`(实验细节不完整),routes.jsonl 侧也会用同 code 标注受影响的路线。

`source.references[]` 元素:


| 字段             | 类型         | 必填   | 说明                                                  |
| ---------------- | ------------ | ------ | ----------------------------------------------------- |
| `reference_id`   | str          | 是     | 被引用的 procedure 标识,如`example_1`、`example_18`   |
| `kind`           | str          | 是     | 常见 `procedure`，也可为 `precursor_preparation`；可扩展 |
| `status`         | str\| null   | 是     | 常见 `resolved`（已解析）/ `missing`（未找到）；可扩展，未知值不视为已解析 |
| `procedure_type` | str          | 否     | 常见 `example` / `preparation` / `step` / `general` 等原文标题类型；可扩展，可能缺键 |
| `pages`          | array\<int\> | 是     | 引用目标所在页码(1-based),可为空数组                  |
| `target`         | object       | **否** | 可选键,仅`status=resolved` 时存在,定位引用落点        |

`source.references[].target` 结构:


| 字段                | 类型            | 必填 | 说明                                                                                                                                  |
| ------------------- | --------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `chunk_id`          | str             | 是   | 引用落点所在 chunk                                                                                                                    |
| `parent_example_id` | str\| null      | 是   | 上级 example 标识,可为 null                                                                                                           |
| `blocks`            | array\<object\> | 是   | 原文 block 级定位,元素含`page_idx`(int,**0-based**,与 pages 差 1)、`block_idx`(int)、`bbox`(array\<number\>, `[x0, y0, x1, y1]` 坐标) |

## 3. routes.jsonl

一行一条**合成路线**记录。

### 3.1 顶层字段


| 字段                 | 类型            | 必填 | 说明                                                         |
| -------------------- | --------------- | ---- | ------------------------------------------------------------ |
| `schema_version`     | str             | 是   | 恒为`route-v1`                                               |
| `route_id`           | str             | 是   | 全局唯一主键,格式见 §1                                       |
| `document_id`        | str             | 是   | 所属文档                                                     |
| `process_kind`       | str             | 是   | `synthesis` / `formulation` / `purification` / `unknown`       |
| `title`              | str / null      | 是   | 路线标题,通常为目标化合物的原文名称,如`"a Crystalline Solid of 8.5-hydrate of ..."`;缺失时为 null |
| `target`             | object          | 是   | 路线目标化合物(见 §3.2)                                      |
| `target_group_id`    | str             | 是   | 目标分组 ID,格式`{目标reaction_id}::target::{序号}`          |
| `target_group_basis` | str             | 是   | 目标化合物分组依据:`canonical_smiles`(按结构)/ `exact_name`(按精确名称)/ `parent_name`(按母体名称) |
| `steps`              | array\<str\>    | 是   | 元素为 `reaction_id`,按文档内记录顺序排列,不保证是执行顺序;目标反应由 `target.reaction_id` 指定,不保证位于数组末尾 |
| `connections`        | array\<object\> | 是   | 步骤间有证据支持的物料承接关系(见 §3.3),不要求两步在 `steps` 数组中相邻;单步路线为空数组 |
| `quality`            | object          | 是   | 质检结果(见 §3.4)                                            |

### 3.2 target


| 字段                 | 类型 | 必填 | 说明                                                                                             |
| -------------------- | ---- | ---- | ------------------------------------------------------------------------------------------------ |
| `target.reaction_id` | str  | 是   | 目标对应的反应,指向 reactions.jsonl 的主键,且一定出现在 `steps` 中;数组位置不代表其目标身份 |
| `target.compound_id` | str / null | 是 | 目标化合物在**该反应** `compounds` 中的局部 ID(如 `c2`);只从源标题识别到目标、没有可引用的抽取产物时为 null |
| `target.name` | str | 否 | `compound_id` 为 null 时提供的源标题目标名称 |
| `target.basis` | str | 否 | `compound_id` 为 null 时为 `source_title`,并在质量问题中标记 `target_from_title_without_extracted_product` |

### 3.3 connections[] 元素


| 字段            | 类型 | 必填   | 说明                                                                                                                                   |
| --------------- | ---- | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `from`          | str  | 是     | 上游步骤的`reaction_id`                                                                                                                |
| `to`            | str  | 是     | 下游步骤的`reaction_id`                                                                                                                |
| `input_id`      | str  | 是     | 承接物料在**to 反应** `compounds` 中的局部 ID(作为输入投入下游)                                                                        |
| `product_id`    | str  | 是     | 承接物料在**from 反应** `compounds` 中的局部 ID(上游产物)                                                                              |
| `basis`         | str  | 是     | 连接判定依据:`canonical_smiles`(上下游结构 SMILES 匹配)/ `material_reference`(依据原文文字引用,如 "the compound 8 obtained by Step 1") |
| `evidence_text` | str  | **否** | 可选键,仅`basis=material_reference` 时出现,为判定所依据的原文文本                                                                      |

### 3.4 quality（内部质量控制）


| 字段                         | 类型            | 必填 | 说明                                                                    |
| ---------------------------- | --------------- | ---- | ----------------------------------------------------------------------- |
| `quality.admission_eligible` | bool            | 是   | 同 §2.5;样本中 155/155 为 false                                        |
| `quality.blocking_issues`    | array\<object\> | 是   | **对象数组**(与 reactions.jsonl 的字符串数组不同,见 §5.2),元素结构见下 |
| `quality.policy_version`     | str             | 是   | 同 §2.5                                                                |

`quality.blocking_issues[]` 元素(按 `code` 不同字段形状不同):


| 字段                     | 类型         | 必填   | 说明                                                                                                                                                                                                                                              |
| ------------------------ | ------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `code`                   | str          | 是     | 可扩展代码，常见值见下文；遇到未知 code 保留并展示通用问题，不丢弃记录 |
| `reaction_ids`           | array\<str\> | 是     | 该问题涉及的`reaction_id` 列表                                                                                                                                                                                                                    |
| `reference_id`           | str          | **否** | 过程引用问题可能携带的原文引用 ID |
| `candidate_reaction_ids` | array\<str\> | **否** | 仅`code=ambiguous_upstream` 时存在,列出候选上游反应                                                                                                                                                                                               |

当前常见 code：`step_quality_blocked`（成员反应未通过质检）、`structure_conflict`（结构冲突）、`telescoped_without_reported_intermediates`（一锅法无中间体报告）、`procedure_reference_missing`（过程引用缺失）、`ambiguous_upstream`（上游来源不唯一）、`connection_material_unavailable`（连接物料不可引用）、`target_from_title_without_extracted_product`（目标仅来自标题）、`disconnected_procedure_steps`（片段未连接）、`terminal_product_coverage_unknown`（终点产物覆盖不明）。过程引用还可能产生其他 `procedure_reference_*` 代码；此列表不是封闭集合。

## 5. 入库注意事项

### 5.1 局部 ID 与联合键

`compound_id` 是 reaction 内局部编号,不可单独作化合物表主键

### 5.2 可选键说明

以下字段为**可选键**:记录中可能完全没有该键(而非值为 null),解析时须用"键是否存在"判断:

- `conditions.yield.reported_values`
- `conditions.yield.range`、`raw`、`approximate`、`reported_ranges`、`basis`
- `source.references`(整个数组)
- `source.references[].procedure_type`
- `source.references[].target`
- `connections[].evidence_text`
- routes `quality.blocking_issues[]` 中的 `reference_id`、`candidate_reaction_ids`
- `target.name`、`target.basis`
- 反应 `source_models` 及化合物附加信息

各表标“是”的键在所属对象存在时提供；允许值为 null 的对象应先判空。不要将“键存在”误读为“对象或数值一定非空”。

### 5.3 页码基准不一致

`source.pages`、`source.references[].pages` 为 **1-based**;`source.references[].target.blocks[].page_idx` 为 **0-based**(同一位置相差 1)。入库时建议统一为一种基准并记录约定。

### 5.4 自由文本字段

`conditions.temperature`、`conditions.time`、`actions[]` 均为**原文级自由文本/词组,未归一化**(温度可能混入操作描述如 `"ice-bath"`、`"refrigerator"`,actions 有大小写与形态变体)。如需数值分析,应在下游另行解析,不要在入库层强转数值类型。

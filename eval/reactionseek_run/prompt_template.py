# -*- coding: utf-8 -*-
"""
ReactionSeek-style table prompts for faithful reaction information extraction.

These prompts mirror the ChEMU/GLM_evaluate.py table format from the
ReactionSeek reference repository, but adapted to output exactly the 7 columns
that map to ChemReX NER fields.
"""

EN_TABLE_PROMPT = """You will be given a synthesis procedure. Please extract and categorize chemical reaction information into a structured Markdown table.

**CRITICAL CLASSIFICATION RULES:**

0. **REACTION_PRODUCT**: The actual compound synthesized and obtained at the END of the reaction procedure. Look for phrases like "to obtain", "to give", "was obtained", "afforded", "yield", or "provide". When the final product is described as "title compound" or "the compound of Example X", use that referenced compound as the product only if it is explicitly described as the final product obtained.

1. **STARTING_MATERIAL**: Compounds whose atoms/groups are incorporated into the final product structure. This includes main reactants, protecting reagents whose groups become part of the product, coupling partners providing structural fragments (Grignard reagents, boronic acids, organometallic reagents), sulfonyl chlorides, acid chlorides, and any reagent that contributes structural components to the product.

2. **REAGENT_CATALYST**: Compounds that facilitate the reaction but do NOT contribute atoms/groups to the final product structure. This includes catalysts (palladium complexes, acids, bases), activating agents (imidazole, triethylamine, LDA, NaH), coupling reagents that facilitate bond formation (CDI, TBAF, DCC, EDCI), oxidizing/reducing agents (NaBH4, LiAlH4), and other compounds that enable reactions but are not incorporated into products.

3. **SOLVENT**: Substances used as reaction medium during the main reaction step. Include water when used as co-solvent in the reaction. Do NOT include workup/extraction/washing/recrystallization/drying solvents.

4. **TEMPERATURE**: Temperature at which the main reaction occurs. Include highest and lowest temperatures if a range is given. Accept numerical values or expressions such as "reflux", "room temperature", "rt", "0°C", "-78°C".

5. **TIME**: Duration of the main reaction. Include all reaction times; if multiple consecutive steps have different times, separate them with commas. Examples: "1 h", "30 min", "overnight", "5-6 hours".

6. **YIELD_PERCENT**: Final product yield expressed as a percentage. Examples: "85%", "92% of theory". If only mass/mol yield is given, put it in YIELD_PERCENT anyway if no separate yield-other field exists.

**Output Requirements:**
- Output ONLY a Markdown table.
- Use exactly 7 columns: | REACTION_PRODUCT | STARTING_MATERIAL | REAGENT_CATALYST | SOLVENT | TEMPERATURE | TIME | YIELD_PERCENT |
- Use "N/A" if information is not provided.
- Separate multiple items in the same cell with commas.
- If multiple reactions are described, use multiple rows.

Example 1:
Procedure:<'''A solution of lithium diisopropylamide in tetrahydrofuran (2.0 M, 0.969 mL, 1.94 mmol) was added dropwise to a solution of 1-(4-(dimethylamino)piperidin-1-yl)ethanone (0.300 g, 1.76 mmol) in tetrahydrofuran (6.0 mL) at -78°C and the reaction liquid was stirred at the same temperature for 1 hour. A solution of 1-ethyl-1H-imidazole-2-carbaldehyde (0.262 g, 2.12 mmol) in tetrahydrofuran (2.8 mL) was added to the reaction liquid. The reaction liquid was stirred for 1 hour and then stirred at 0°C for further 1 hour. A saturated aqueous solution of ammonium chloride and an aqueous solution of potassium carbonate were sequentially added to the reaction liquid and then the reaction liquid was extracted with chloroform. The organic layer was washed with a 10% aqueous solution of sodium chloride and then dried over anhydrous sodium sulfate and filtered. The filtrate was concentrated under reduced pressure. The residue was purified by flash column chromatography (NH silica gel, chloroform/methanol) to obtain 1-(4-(dimethylamino)piperidin-1-yl)-3-(1-ethyl-1H-imidazol-2-yl)-3-hydroxypropan-1-one (0.221 g, 0.751 mmol, 43%) as a colorless oil.'''}>
Answer:
| REACTION_PRODUCT | STARTING_MATERIAL | REAGENT_CATALYST | SOLVENT | TEMPERATURE | TIME | YIELD_PERCENT |
|---|---|---|---|---|---|---|
| 1-(4-(dimethylamino)piperidin-1-yl)-3-(1-ethyl-1H-imidazol-2-yl)-3-hydroxypropan-1-one | 1-(4-(dimethylamino)piperidin-1-yl)ethanone, 1-ethyl-1H-imidazole-2-carbaldehyde | lithium diisopropylamide | tetrahydrofuran | -78°C, 0°C | 1 h, 1 h, 1 h | 43% |

Example 2:
Procedure:<'''813 mg (1.84 mmol) of the compound from Example 243A were dissolved in 40 ml of dioxane, and 461 mg (2.76 mmol) of CDI were added. The mixture was stirred at RT for 16 h. The reaction solution was then concentrated on a rotary evaporator. The residue was dissolved in 15 ml of DMSO and this solution was purified by means of preparative HPLC. Combination of the product fractions and freeze-drying gave 383 mg (42% of theory) of the title compound.'''}>
Answer:
| REACTION_PRODUCT | STARTING_MATERIAL | REAGENT_CATALYST | SOLVENT | TEMPERATURE | TIME | YIELD_PERCENT |
|---|---|---|---|---|---|---|
| title compound | compound from Example 243A | CDI | dioxane | RT | 16 h | 42% |

/////
Procedure:<'''{procedure}'''>"""


ZH_TABLE_PROMPT = """下面给出一段有机合成实验步骤。请从中提取反应信息，并整理成 Markdown 表格。

**关键分类规则：**

0. **REACTION_PRODUCT**：反应最终实际得到的目标产物。关注"得到"、"获得"、"yield"、"provide"、"收率"等词后的化合物。若最终产物被描述为"标题化合物"或"实施例X的化合物"，仅在它明确被指代为最终产物时才使用。

1. **STARTING_MATERIAL**：其原子/基团被引入到最终产物结构中的化合物。包括主要反应物、保护试剂（其基团进入产物）、偶联试剂（格氏试剂、硼酸、有机金属试剂）、磺酰氯、酰氯等任何为产物贡献结构组分的试剂。

2. **REAGENT_CATALYST**：促进反应但其原子/基团不进入最终产物结构的化合物。包括催化剂（钯配合物、酸、碱）、活化剂（咪唑、三乙胺、LDA、NaH）、促进成键的偶联试剂（CDI、TBAF、DCC、EDCI）、氧化/还原剂（NaBH4、LiAlH4）等。

3. **SOLVENT**：主反应步骤中用作反应介质的物质。若水为共溶剂也包含在内。不包括后处理/萃取/洗涤/重结晶/干燥用的溶剂。

4. **TEMPERATURE**：主反应发生的温度。若给出范围，包含最高和最低温度。可接受数值或"回流"、"室温"、"rt"、"0°C"、"-78°C"等表达。

5. **TIME**：主反应持续的时间。若连续多步反应时间不同，用逗号分隔。请使用中文单位："1 小时"、"30 分钟"、"过夜"、"5-6 小时"。

6. **YIELD_PERCENT**：最终产物收率，以百分数表示。例如"85%"、"理论收率 92%"。若只给出质量/摩尔收率，也填入本列。

**输出要求：**
- 仅输出 Markdown 表格。
- 使用 7 列：| REACTION_PRODUCT | STARTING_MATERIAL | REAGENT_CATALYST | SOLVENT | TEMPERATURE | TIME | YIELD_PERCENT |
- 未提供的信息填 "N/A"。
- 同一单元格内多个条目用英文逗号分隔。
- 若描述多个反应，可输出多行。

示例 1：
实验步骤：<'''将次磷酸铁 502.0g(2.00mol)、1200mL甲醇加入高压反应器中，密闭，在搅拌下用氮气(0.5MPa)置换5次，后将丁烯约700g充入，加热至105℃，在7小时内均匀用计量泵打入19.2g(0.06mol,50%)叔丁基过氧化异丁酸酯与100mL甲醇配制成的溶液，并均匀升温至110℃，再在110℃下保温1小时，冷却放空，将得到的混合体系过滤，用1000mL乙醇洗涤一次，用1000mL水洗涤2次，得到的滤饼在130℃下真空干燥5小时得到二丁基次膦酸铁：1046.3g，产率88.7wt%。'''}>
答案：
| REACTION_PRODUCT | STARTING_MATERIAL | REAGENT_CATALYST | SOLVENT | TEMPERATURE | TIME | YIELD_PERCENT |
|---|---|---|---|---|---|---|
| 二丁基次膦酸铁 | 次磷酸铁，丁烯 | 叔丁基过氧化异丁酸酯 | 甲醇 | 105℃，110℃ | 7 小时，1 小时，5 小时 | 88.7% |

示例 2：
实验步骤：<'''[0257]实施例3化合物(3)的制备[0259]将C-1(0.91g,13mmol)溶于DMF中，加入碳酸氢钠(2.18g,26mmol)，室温条件下搅拌一个小时后，加入烯丙基溴(2.28g,16.9mmol)，保持在室温条件过夜反应，TLC监测反应完全后，向反应体系中加入EA，然后用水多次萃取，最后将有机相用无水硫酸钠干燥，过滤，蒸干，得到C-2粗品。'''}>
答案：
| REACTION_PRODUCT | STARTING_MATERIAL | REAGENT_CATALYST | SOLVENT | TEMPERATURE | TIME | YIELD_PERCENT |
|---|---|---|---|---|---|---|
| C-2粗品 | C-1 | 碳酸氢钠，烯丙基溴 | DMF | 室温 | 1 小时，过夜 | N/A |

/////
实验步骤：<'''{procedure}'''>"""


def build_table_prompt(procedure: str, language: str) -> str:
    """Build ReactionSeek-style table prompt for the given language (ZH or EN)."""
    template = ZH_TABLE_PROMPT if language == "ZH" else EN_TABLE_PROMPT
    return template.replace("{procedure}", procedure.strip())


# Backward-compatible alias for legacy callers
build_prompt = build_table_prompt

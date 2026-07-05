# NER_ZH
以下是一段有机反应的描述。分析文本并将以下字段提取到 Markdown JSON 代码块中：reaction_product, starting_material, reagent_catalyst, solvent, temperature, time, yield_percent。**仅输出 JSON**
###实验步骤：{{TEXT}}

# Action_ZH
以下是一段有机反应的描述。分析文本并按先后顺序提取其中的**原子操作动词**（例如加入、纯化等），用**英文逗号**分隔。**仅输出单行动词序列**。
###实验步骤：{{TEXT}}

# NER_EN
Below is a description of an organic reaction. Analyze text and extract the following fields into a Markdown JSON code block:reaction_product, starting_material, reagent_catalyst, solvent, temperature, time, yield_percent.**Output ONLY the JSON**
###Procedure:{{TEXT}}

# Action_EN
Below is a description of an organic reaction. Analyze text and extract the **atomic action verbs**(e.g., add, purify) in chronological order, separated by **English commas**.**Output ONLY a single line of verbs**.
###Procedure:{{TEXT}}
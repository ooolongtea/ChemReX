# Swern Oxidation — Coreference Example

## 1. Simplified input text

A condensed experimental paragraph. Compound names are replaced by identifiers
**1a** (starting material) and **2a** (product) to exercise coreference resolution.
All 7 extractable elements are present: `starting_material`, `reaction_product`,
`reagent_catalyst`, `solvent`, `temperature`, `time`, `yield_percent`.

> To a solution of oxalyl chloride (1.2 eq) in CH2Cl2 at -78 °C under argon is
> added dropwise DMSO (2.4 eq) in CH2Cl2. After stirring for 1 hour at -78 °C,
> a solution of **1a** (1.0 eq) in CH2Cl2 is added dropwise. After stirring for
> 2 hours at -78 °C, triethylamine (5.0 eq) is added dropwise, and the mixture is
> stirred at -78 °C for 1 hour, then warmed to 25 °C and stirred overnight. The
> reaction is partitioned between CH2Cl2 and 1 N HCl, and the organic layer is
> washed with water, dried over MgSO4, and concentrated in vacuo to afford **2a**
> (95%) as a yellow oil.

Expected NER fields:

| field | value |
|---|---|
| starting_material | 1a |
| reaction_product | 2a |
| reagent_catalyst | oxalyl chloride; DMSO; triethylamine |
| solvent | CH2Cl2 |
| temperature | -78 °C; 25 °C |
| time | 1 hour; 2 hours; 1 hour; overnight |
| yield_percent | 95% |

## 2. SMILES and coreference mapping

The text uses identifiers (1a, 2a) that OPSIN cannot convert to SMILES, so the
standardizer marks them as `ID`. Coreference resolution maps them to the SMILES
recovered from the reaction image (MolDetect pairs each label with its drawn
structure; MolScribe reads the structure).

| identifier | compound name | SMILES |
|---|---|---|
| 1a | 5-phenyl-1-pentanol | `OCCCCCc1ccccc1` |
| 2a | 5-phenylpentanal | `O=CCCCCc1ccccc1` |

Reaction expression (1a >> 2a):

```
OCCCCCc1ccccc1>>O=CCCCCc1ccccc1
```

Supporting reagent/solvent SMILES (resolved directly from names by OPSIN, not via coref):

| name | standard name | SMILES |
|---|---|---|
| oxalyl chloride | oxalyl chloride | `O=C(Cl)C(=O)Cl` |
| DMSO | dimethyl sulfoxide | `CS(C)=O` |
| triethylamine | triethylamine | `CCN(CC)CC` |
| CH2Cl2 | dichloromethane | `ClCCl` |

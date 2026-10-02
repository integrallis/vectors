# Band/integer dispatch — model-level check (pre-registration 2)

vectors=a475c3b48c443a59cce0f4cccef8e343de13c935 models=167a8abdd662af8d89a821a1bf2d23980e1092c5 window=cb2f42624d6cb75e8caa9fbe1715233ea7148eb2

## 1. Prefill tok/s (context 2048)

| rep | arm | prompt tokens | tok/s | logit checksum | routing |
|---:|---|---:|---:|---|---|
| 1 | dispatch | 2040 | 43.22 | -8357.73625 | Q4_K:int 0/0 band 46080/1468800 Q6_K:int 120/2880 band 7560/241920 Q8_0:int 0/0 band 0/0 (mode=dispatch, int rows=2880, band rows=1710720) |
| 1 | integer | 2040 | 12.77 | 98.1167452 | Q4_K:int 46080/1468800 band 0/0 Q6_K:int 7680/244800 band 0/0 Q8_0:int 0/0 band 0/0 (mode=integer, int rows=1713600, band rows=0) |
| 2 | dispatch | 2040 | 35.05 | -8357.73625 | Q4_K:int 0/0 band 46080/1468800 Q6_K:int 120/2880 band 7560/241920 Q8_0:int 0/0 band 0/0 (mode=dispatch, int rows=2880, band rows=1710720) |
| 2 | integer | 2040 | 13.59 | 98.1167452 | Q4_K:int 46080/1468800 band 0/0 Q6_K:int 7680/244800 band 0/0 Q8_0:int 0/0 band 0/0 (mode=integer, int rows=1713600, band rows=0) |
| 3 | dispatch | 2040 | 38.86 | -8357.73625 | Q4_K:int 0/0 band 46080/1468800 Q6_K:int 120/2880 band 7560/241920 Q8_0:int 0/0 band 0/0 (mode=dispatch, int rows=2880, band rows=1710720) |
| 3 | integer | 2040 | 14.53 | 98.1167452 | Q4_K:int 46080/1468800 band 0/0 Q6_K:int 7680/244800 band 0/0 Q8_0:int 0/0 band 0/0 (mode=integer, int rows=1713600, band rows=0) |
| 4 | dispatch | 2040 | 43.63 | -8357.73625 | Q4_K:int 0/0 band 46080/1468800 Q6_K:int 120/2880 band 7560/241920 Q8_0:int 0/0 band 0/0 (mode=dispatch, int rows=2880, band rows=1710720) |
| 4 | integer | 2040 | 14.17 | 98.1167452 | Q4_K:int 46080/1468800 band 0/0 Q6_K:int 7680/244800 band 0/0 Q8_0:int 0/0 band 0/0 (mode=integer, int rows=1713600, band rows=0) |
| 5 | dispatch | 2040 | 31.46 | -8357.73625 | Q4_K:int 0/0 band 46080/1468800 Q6_K:int 120/2880 band 7560/241920 Q8_0:int 0/0 band 0/0 (mode=dispatch, int rows=2880, band rows=1710720) |
| 5 | integer | 2040 | 13.79 | 98.1167452 | Q4_K:int 46080/1468800 band 0/0 Q6_K:int 7680/244800 band 0/0 Q8_0:int 0/0 band 0/0 (mode=integer, int rows=1713600, band rows=0) |

**PREFILL: median integer 13.79 tok/s, dispatch 38.86 tok/s, change +181.8% (min/max integer 12.77/14.53, dispatch 31.46/43.63) -> PASS (gate >= +10%)**

## 2. Greedy continuations, first 20 squad-v2-dev cases, max 64 tokens

**CONTINUATIONS base-arm (gate): identical 17/20 -> FAIL (gate >= 19/20)**  integer-arm completion tokens mean 2.8 (min 1, max 7), stop reasons ['EOS']

  note: base-arm outputs are short (one-word instruction), so identity is tested over few tokens; see the open variant.

  - 5737432bc3c5551400e51e9b: first divergence at token 0; integer="Newton" dispatch="unanswerable"
  - 5705f09e75f01819005e77a4: first divergence at token 0; integer="unanswerable" dispatch="taxes"
  - 57266193dd62a815002e832e: first divergence at token 0; integer="answerable" dispatch="wave speeds"

**CONTINUATIONS open (supplementary, not gating): identical 10/20 -> reported**  integer-arm completion tokens mean 36.4 (min 9, max 64), stop reasons ['EOS', 'MAX_TOKENS']

  - 5ad247b0d7d075001a428b45: first divergence at token 4; integer="The document does not specify a tax system that has no impact on income inequality. It discusses how a progressive tax system can either increase or decrease inequality, depending on the level of the top tax rate and its application to social spending. Therefore, based on the information provided, there is no mentioned system that has no impact on" dispatch="The document does not provide information on a tax system that has no impact on income inequality. It discusses how progressive tax systems can either increase or decrease income inequality, depending on the level of the top tax rate and its application to social spending. Therefore, based on the given information, there is no mentioned system that has no"
  - 5705f09e75f01819005e77a4: first divergence at token 1; integer="The document does not provide specific details on what else the Californios were dissatisfied with beyond inequitable taxes and land laws." dispatch="The Californios were dissatisfied with inequitable taxes in addition to land laws."
  - 5a2c1397bfd06b001a5ae9c9: first divergence at token 4; integer="The document does not mention that Alec Shelbrooke proposed payments of benefits should never be made on any specific type of payment method. It only states that he was proposing the payments of benefits and tax credits on a \"Welfare Cash Card\", which could be used to buy only \"essentials\"." dispatch="The document does not specify that Alec Shelbrooke proposed payments of benefits should never be made on any specific type of card or method. It mentions his proposal for the payments of benefits and tax credits on a \"Welfare Cash Card,\" which could be used to buy only \"essentials.\" However, the document does not"
  - 5ad4d2105b96ef001a10a1c7: first divergence at token 0; integer="Killer T cells, not Killer B cells, are responsible for killing cells infected with viruses or other pathogens, or those that are damaged or dysfunctional. The document does not mention anything about Killer B cells." dispatch="The document provided does not contain information about Killer B cells or what they kill. It focuses on Killer T cells and their function in killing infected or damaged cells. Therefore, based on the given document, there is no information to answer the question about Killer B cells."
  - 572fd1c4947a6a140053cd02: first divergence at token 30; integer="The final stage of a bill in the Scottish Parliament is Stage 3, which is considered at a meeting of the whole Parliament. This stage includes a general debate on the final form of the bill and a final vote on the bill. Opposition members may also introduce \"wrecking amendments\" to hinder further progress." dispatch="The final stage of a bill in the Scottish Parliament is Stage 3, which is considered at a meeting of the whole Parliament. This stage includes a consideration of amendments as a general debate and a final vote on the bill. Opposition members may also introduce \"wrecking amendments\" to hinder further progress and cause the bill"
  - 5a834caae60761001a2eb53d: first divergence at token 38; integer="The document suggests that the combination of hermaphroditism and early reproduction enables small populations to grow at an explosive rate. However, it does not provide specific quantitative data on how quickly the plankton populations grow. Therefore, based on the information given, we can only conclude that the growth is rapid due to these" dispatch="The document suggests that the combination of hermaphroditism and early reproduction enables small populations to grow at an explosive rate. However, it does not provide specific quantitative data on how quickly plankton populations grow due to these factors. The information given is qualitative, indicating the potential for rapid population growth rather than precise growth"
  - 5a7b3dd921c2de001afe9dfa: first divergence at token 25; integer="The document does not provide information on the Filipino population percentage in Fresno in 1970. It only gives the racial breakdown for 2010. Therefore, the percentage of Filipino residents in Fresno in 1970 cannot be determined from the available data." dispatch="The document does not provide information on the Filipino population percentage in Fresno in 1970. It only gives the racial breakdown for the year 2010. Therefore, the percentage of Filipino residents in Fresno in 1970 cannot be determined from the available data."
  - 5a63835a68151a001a92232e: first divergence at token 5; integer="The information provided does not include details about amendments to the United Kingdom Parliament. It specifically discusses the amendment process for the Victorian Constitution, which is governed by the Parliament of Victoria, not the United Kingdom Parliament. Therefore, based on the given document, there is no indication of which group can amend the United Kingdom Parliament." dispatch="The information provided does not contain details about any group that can amend the United Kingdom Parliament. It specifically discusses the amendment process for the Victorian Constitution within Australia, relating to the Parliament of Victoria and its relationship to the 1855 colonial constitution. Therefore, based on the given document, there is no indication of a group capable"
  - 572681c1dd62a815002e8797: first divergence at token 22; integer="Platyctenida use their pharynx to evert and function as a muscular \"foot\" for them to cling to and creep on surfaces." dispatch="Platyctenida use their pharynx to evert and function as a muscular \"foot\" for clinging to and creeping on surfaces."
  - 5ad158c0645df0001a2d182a: first divergence at token 1; integer="The Court of Justice stated that Austria was **not** allowed to hold places in Austrian universities exclusively for Austrian students. The case in question is Commission v Austria." dispatch="The document does not state that the Court allowed Austria to hold places in Austrian schools exclusively for Austrian students. In fact, it states the opposite: \"In Commission v Austria the Court held that Austria was not entitled to restrict places in Austrian universities to Austrian students to avoid 'structural, staffing and financial problems' if (main"

PROMPT PARITY with the Models runner: SKIPPED (ADAPTER_DIR not set)

## Observability problems

- none

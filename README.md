# Rules2UD [![Launch Binder](http://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jerrybonnell/Rules2UD/main?filepath=demo.ipynb)

Rule-based Adornment of Modern Historical Japanese Corpora using Accurate Universal Dependencies

```
python modify_conllu.py [INFILE] [APPLY_RULES] [MODEL] [OUT_DIR] [VERBOSE]

-INFILE
    input file pointing to a list of raw sentences to use for processing
    (see sents.txt). sentences can also be redirected through stdin or
    supplied as an XML where sentences are provided as <s> tags

-APPLY_RULES
    a boolean (True/False) for whether to do rule application

-MODEL
    annotation model for supplying UD. can be "ginza" or "esupar"

-OUT_DIR
    output directory for the output CoNLL-U file. defaults to a
    directory called "out"

-VERBOSE
    should verbose printing be allowed? (True/False)
```

## Prerequisites

Install all required packages using the supplied `requirements.txt` file.

```
pip install -r requirements.txt
```

## Demo usage

```
echo "社會の發達に從ふて、" | python modify_conllu.py True "ginza" "out"
log : {'ふ': [(7, 7)], '會': [(1, 1)], '從': [(6, 6)], '發': [(3, 3)]}

no corruptions found in FORM ✅
# text = 社會の發達に從ふて、
1       社會      社会      NOUN    名詞-普通名詞-一般      _       3       nmod    _
SpaceAfter=No|BunsetuBILabel=B|BunsetuPositionType=SEM_HEAD|NP_B|Reading=シャカイ
2       の       の       ADP     助詞-格助詞  _       1       case    _       SpaceAfter=No|BunsetuBILabel=I|BunsetuPositionType=SYN_HEAD|Reading=ノ
3       發達      発達      NOUN    名詞-普通名詞-サ変可能    _       5       obl     _
SpaceAfter=No|BunsetuBILabel=B|BunsetuPositionType=SEM_HEAD|NP_B|Reading=ハッタツ
4       に       に       ADP     助詞-格助詞  _       3       case    _       SpaceAfter=No|BunsetuBILabel=I|BunsetuPositionType=SYN_HEAD|Reading=ニ
5       從ふ      従う      VERB    動詞-一般   _       0       root    _
SpaceAfter=No|BunsetuBILabel=B|BunsetuPositionType=ROOT|Inf=五段-ワア行,連用形-ウ音便|Reading=シタゴウ
6       て       て       SCONJ   助詞-接続助詞 _       5       mark    _       SpaceAfter=No|BunsetuBILabel=I|BunsetuPositionType=SYN_HEAD|Reading=テ
7       、       、       PUNCT   補助記号-読点 _       5       punct   _       SpaceAfter=No|BunsetuBILabel=I|BunsetuPositionType=CONT|Reading=、
```

Or redirect an entire file:

```
python modify_conllu.py True "ginza" "out" False < sents.txt
```

A Binder link is supplied at the top of this README for a quick demo on how to use the tool.

## Inspecting/modifying the ruleset

Ruleset descriptions are provided as JSON in `rules/`. New rule sets can be introduced simply by creating new JSON files in this folder. Use the existing JSON  as template for forming new rulesets.

To extend rules within a ruleset, form new key-value pairs in the `"rules"` dictionary. For instance, adding `"踰ゆ": "越えゆ"` to the dictionary in `"rules"` in `basic_replace_rules.json` will make `modify_conllu.py` aware of this new rule.

Some management of the ruleset is needed. Before adding a new rule, confirm that it brings the desired output when running the sentence through GiNZA. If not, the rule should be modified accordingly or not included. A diagnostic check is performed at the end of the program to check for issues in the FORM column as a result of applying malformed rules.

## Evaluation scripts

Evaluation scripts for interfacing with UDPipe and generating the figures shown in the paper are not included here. There are copyright issues with the source corpus that prevent us from possibly revealing any training data used and many of our scripts depend on the machines made available to us at the time of this research.

However, we may still be able to produce some of these scripts depending on your needs. Please reach out to us.

## Contact

For bugs or questions related to the code, please raise a GitHub issue. For any other questions, please reach out to [j.bonnell@miami.edu](mailto:j.bonnell@miami.edu) and/or [m.ogihara@miami.edu](mailto:m.ogihara@miami.edu).


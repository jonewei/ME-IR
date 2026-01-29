import json
from tqdm import tqdm
from retrieval.approach0_hash import Approach0HashIndex


def load_formulas(formula_path):
    """
    Load formulas.jsonl
    """
    formulas = []
    with open(formula_path, "r", encoding="utf-8") as f:
        for line in f:
            formulas.append(json.loads(line))
    return formulas


def build_index(formula_path: str) -> Approach0HashIndex:
    """
    Build structural index from formulas.jsonl
    """
    index = Approach0HashIndex()
    formulas = load_formulas(formula_path)

    for f in tqdm(formulas, desc="Building structural index"):
        index.add(f)

    return index

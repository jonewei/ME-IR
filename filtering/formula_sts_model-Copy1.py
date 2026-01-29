import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FormulaSTSModel:
    """
    Binary semantic similarity classifier for formulas.
    """

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.92,
        device: str = None,
        max_length: int = 128,
    ):
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_prob(self, formula_a: str, formula_b: str) -> float:
        inputs = self.tokenizer(
            formula_a,
            formula_b,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        # assume label 1 = similar
        return probs[0, 1].item()

    def is_similar(self, formula_a: str, formula_b: str) -> bool:
        return self.predict_prob(formula_a, formula_b) >= self.threshold

#真正的 STS 二分类推理模型
# 不是 cosine trick

# 是真正 pairwise binary classifier

# 你可以 fine-tune 成 formula-BERT-STS
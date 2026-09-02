"""Coding-potential task: probability that each sequence is protein-coding."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader


class CodingPotentialHandler(TaskHandler):
    task = Task.CODING_POTENTIAL

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)  # raises WeightsNotFoundError if absent
        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForSequenceClassification
        from fungidna.data.tokenizer import DualTokenizer

        bpe_path = ckpt.parent / "tokenizer.model"
        tok = DualTokenizer(str(bpe_path))
        model = FungiDNAForSequenceClassification(FungiDNAConfig(), num_classes=2)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()

        predictions = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                ids = torch.tensor([tok.encode_bpe(seq, add_cls=True)], dtype=torch.long)
                logits = model(ids)
                prob = torch.softmax(logits, dim=1)[0, 1].item()  # class 1 = coding
                predictions[header] = {
                    "coding_probability": prob,
                    "label": "coding" if prob >= 0.5 else "non-coding",
                }
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"predictions": predictions})

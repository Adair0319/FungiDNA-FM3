"""Splice-site task: classify candidate splice motifs as donor/acceptor/non-site.

Each candidate is a GT/GC (donor) or AG (acceptor) dinucleotide. The model
classifies the surrounding ±200 bp window into 3 classes:
0 = donor, 1 = acceptor, 2 = non-site.
"""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader

_SPLICE_MOTIFS = {"GT", "GC", "AG"}
_WINDOW_HALF = 200           # ±200 bp around the candidate site → 401 bp window
_NUM_CLASSES = 3             # donor / acceptor / non-site


class SpliceSiteHandler(TaskHandler):
    task = Task.SPLICE_SITE

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)  # raises WeightsNotFoundError if absent
        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForSequenceClassification
        from fungidna.data.tokenizer import DualTokenizer

        bpe_path = ckpt.parent / "tokenizer.model"
        tok = DualTokenizer(str(bpe_path))
        model = FungiDNAForSequenceClassification(FungiDNAConfig(), num_classes=_NUM_CLASSES)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()

        predictions = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                seq = seq.upper()
                hits = []
                for i in range(len(seq) - 1):
                    motif = seq[i:i + 2]
                    if motif not in _SPLICE_MOTIFS:
                        continue
                    window = seq[max(0, i - _WINDOW_HALF): i + _WINDOW_HALF + 1]
                    ids = torch.tensor([tok.encode_bpe(window, add_cls=True)], dtype=torch.long)
                    logits = model(ids)
                    probs = torch.softmax(logits, dim=1)[0]
                    cls = int(probs.argmax(dim=0).item())
                    hits.append({"pos": i, "motif": motif, "class": cls,
                                 "prob": float(probs[cls].item())})
                predictions[header] = hits
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"predictions": predictions})

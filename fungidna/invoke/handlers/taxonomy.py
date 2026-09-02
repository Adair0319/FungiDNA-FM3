"""Taxonomy task: predict a sequence's dominant phylum by majority vote over windows.

The backbone is the same pretrained StripedMambaBackbone; a small inline MLP head
sits on top (768 -> 256 -> num_classes). Default rank is phylum with 6 classes.
Only the default phylum rank is produced; other ranks can be enabled once their
heads are shipped by swapping ``_NUM_CLASSES`` and the ``rank`` field below.
"""
import torch
import torch.nn as nn

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader

_WINDOW_BP = 10_000          # slide 10 kb windows...
_STEP_BP = 10_000            # ...with a 10 kb stride (contiguous non-overlapping chunks)
_NUM_CLASSES = 6             # phylum count for the default rank
_DEFAULT_RANK = "phylum"


class _TaxonomyClassifier(nn.Module):
    """Backbone + inline MLP head, used to load a whole-model checkpoint."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids)
        pooled = hidden.mean(dim=1)
        return self.head(pooled)


class TaxonomyHandler(TaskHandler):
    task = Task.TAXONOMY

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)  # raises WeightsNotFoundError if absent
        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        bpe_path = ckpt.parent / "tokenizer.model"
        tok = DualTokenizer(str(bpe_path))

        config = FungiDNAConfig()
        backbone = StripedMambaBackbone(config)
        head = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, _NUM_CLASSES),
        )

        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        top_keys = list(sd.keys())
        if top_keys and any(k.startswith(("backbone.", "head.")) for k in top_keys):
            _TaxonomyClassifier(backbone, head).load_state_dict(sd, strict=False)
        else:
            # Bare backbone state dict (keys such as "embedding.*", "layers.*").
            backbone.load_state_dict(sd, strict=False)
        backbone.eval()

        labels = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                votes = []
                for start in range(0, len(seq), _STEP_BP):
                    window = seq[start:start + _WINDOW_BP]
                    if not window:
                        continue
                    ids = torch.tensor([tok.encode_bpe(window, add_cls=True)], dtype=torch.long)
                    logits = head(backbone(ids).mean(dim=1))
                    votes.append(int(logits.argmax(dim=1).item()))
                if votes:
                    # Lowest class index wins ties, keeping the vote deterministic.
                    labels[header] = max(range(_NUM_CLASSES), key=votes.count)
                else:
                    labels[header] = None
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"rank": _DEFAULT_RANK, "labels": labels})

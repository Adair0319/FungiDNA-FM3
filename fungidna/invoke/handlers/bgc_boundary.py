"""BGC-boundary task: predict left/right distance (bp) to the BGC boundaries."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader


class BGCBoundaryHandler(TaskHandler):
    task = Task.BGC_BOUNDARY

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)  # raises WeightsNotFoundError if absent
        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForBGCBoundary
        from fungidna.data.tokenizer import DualTokenizer

        bpe_path = ckpt.parent / "tokenizer.model"
        tok = DualTokenizer(str(bpe_path))
        model = FungiDNAForBGCBoundary(FungiDNAConfig(), num_outputs=2)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        model.load_state_dict(sd, strict=False)
        model.eval()

        boundaries = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                ids = torch.tensor([tok.encode_bpe(seq, add_cls=True)], dtype=torch.long)
                out = model(ids)
                left_bp, right_bp = (float(v) for v in out[0])
                boundaries[header] = {"left_bp": left_bp, "right_bp": right_bp}
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"boundaries": boundaries})

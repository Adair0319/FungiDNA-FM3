"""Representation task: embed each sequence with the pretrained backbone."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader


class RepresentationHandler(TaskHandler):
    task = Task.REPRESENTATION

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        ckpt = self.loader.resolve(self.task.value)  # raises WeightsNotFoundError if absent
        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        # tokenizer model path comes from the checkpoint's sibling dir in real use;
        # kept minimal here — replace with a real tokenizer path when weights ship.
        bpe_path = ckpt.parent / "tokenizer.model"
        tok = DualTokenizer(str(bpe_path))
        model = StripedMambaBackbone(FungiDNAConfig())
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=False)
        model.eval()

        embeddings = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                ids = torch.tensor([tok.encode_bpe(seq, add_cls=True)], dtype=torch.long)
                hidden = model(ids)
                embeddings[header] = hidden.mean(dim=1).squeeze(0).tolist()
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"embeddings": embeddings, "dim": model.config.hidden_size})

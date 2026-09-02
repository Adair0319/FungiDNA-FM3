import unittest
from pathlib import Path

from fungidna.invoke.request import Task, StructuredRequest
from fungidna.invoke.handlers.base import WeightsNotFoundError
from fungidna.invoke.handlers.representation import RepresentationHandler
from fungidna.invoke.handlers.taxonomy import TaxonomyHandler
from fungidna.invoke.handlers.coding_potential import CodingPotentialHandler
from fungidna.invoke.handlers.splice_site import SpliceSiteHandler
from fungidna.invoke.handlers.bgc_boundary import BGCBoundaryHandler


class TestHandlersRequireWeights(unittest.TestCase):
    def _req(self, task):
        return StructuredRequest(task=task, file_path=Path("/tmp/x.fasta"))

    def test_all_handlers_raise_without_weights(self):
        for handler_cls, task in [
            (RepresentationHandler, Task.REPRESENTATION),
            (TaxonomyHandler, Task.TAXONOMY),
            (CodingPotentialHandler, Task.CODING_POTENTIAL),
            (SpliceSiteHandler, Task.SPLICE_SITE),
            (BGCBoundaryHandler, Task.BGC_BOUNDARY),
        ]:
            with self.assertRaises(WeightsNotFoundError, msg=task.value):
                handler_cls().run(self._req(task))

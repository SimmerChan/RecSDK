from typing import List

import tensorflow as tf
from tensorflow import Operation, Graph

from mx_rec.util.log import logger
from mx_rec.graph.slicers import LookupSubgraphSlicer, OrphanLookupKeySlicer


class LookupSubgraphSlicerHook(tf.estimator.SessionRunHook):
    def __init__(self, op_types: List[Operation], full_graph: Graph = None) -> None:
        super().__init__()
        self._op_types = op_types
        self._full_graph = full_graph

    def begin(self) -> None:
        slicer = LookupSubgraphSlicer(self._op_types, self._full_graph)

        logger.info("Starts to summarize sliceable specific operations in lookup subgraph!")
        slicer.summarize()

        logger.info("Starts to slice specific operations and their corresponding minimum dependency graphs!")
        slicer.slice()


class OrphanLookupKeySlicerHook(tf.estimator.SessionRunHook):
    def __init__(self, full_graph: Graph = None) -> None:
        super().__init__()
        self._full_graph = full_graph

    def begin(self) -> None:
        slicer = OrphanLookupKeySlicer(self._full_graph)

        logger.info("Starts to summarize sliceable orphan lookup keys!")
        slicer.summarize()

        logger.info("Starts to slice orphan lookup keys and their corresponding minimum dependency graphs!")
        slicer.slice()

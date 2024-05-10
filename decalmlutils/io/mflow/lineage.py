import base64
import logging
import tempfile
from functools import lru_cache

import graphviz
import requests
from beartype import beartype
from beartype.typing import List, Tuple

from decalmlutils.io.mflow.artifacts import ArtifactMatchers
from decalmlutils.io.mflow.flows import UPSTREAM_PIPELINE_LINKING_PREFIX
from decalmlutils.io.mflow.runs import _process_run
from decalmlutils.io.mflow.tasks import get_last_finished_task

logger = logging.getLogger(__name__)


class MetaflowDag:
    def __init__(self, img_format: str = "png", view: bool = False):
        """
        Instantiate the class and use methods individually. Usage:

        ```
        md = MetaflowDag(img_formt="png", view=False)

        # Option 1 - lineage_string
        md.from_lineage(lineage_string="A->B->C")
        # Option 2 - from_last_run
        md.from_last_run(last_run="TrainModelPipeline/3160")

        # view the dag
        md.dag
        md.dag.save("dag.png")  # save dag to filename dag.png

        # dot source representation of the dag
        md.source
        ```

        Args:
            img_format (str, optional): path to save pdf. Defaults to None i.e. won't be rendered.
            view (bool, optional): automatically open the pdf. Defaults to False.

        """
        self.img_format = img_format
        self.view = view
        self.dag: graphviz.Digraph = None
        self.source: str = None
        self.bytes: bytes = None

    def parse_kwargs(self, **kwargs):
        self.img_format: str = kwargs.get("img_format", self.img_format)
        self.view: bool = kwargs.get("view", self.view)

    @beartype
    def from_lineage(self, lineage_string: str, **kwargs):
        """
        Construct a dag using the lineage object returned by get_artifact_from_lineage.

        Args:
            lineage_string (str): lineage representation returned by get_artifact_from_lineage
            savefile (str, optional): path to save pdf. Defaults to None i.e. won't be rendered.
            view (bool, optional): automatically open the pdf. Defaults to False.

        Returns:
            str: graphViz's Digraph raw dot representation which can be rendered using the following code:
                ```
                # https://graphviz.readthedocs.io/en/stable/manual.html#using-raw-dot
                from graphviz import Source

                dag = MetaflowDag().from_lineage(...)
                Source(dag, img_format="png").render(view=True)
                ```
        """
        self.parse_kwargs(**kwargs)
        self.dag_edges = self.lineage_to_edges(lineage_string)
        self.source = self.create_graph(
            self.dag_edges, img_format=self.img_format, view=self.view
        )
        return self.source

    @lru_cache()
    @beartype
    def from_last_run(self, last_run: str, **kwargs):
        """
        Construct a dag consisting of all upstream flow dependencies.

        Args:
            last_run (str): pathspec of the last run in the dag
            img_format (str, optional): path to save pdf. Defaults to None i.e. won't be rendered.
            view (bool, optional): automatically open the pdf. Defaults to False.

        Returns:
            str: graphViz's Digraph raw dot representation which can be rendered using the following code:
                ```
                # https://graphviz.readthedocs.io/en/stable/manual.html#using-raw-dot
                from graphviz import Source

                dag = MetaflowDag().from_lineage(...)
                Source(dag, img_format="png").render(view=True)
                ```
        """
        logger.info("Creating a lineage DAG. This may take a while...")
        self.parse_kwargs(**kwargs)
        self.dag_edges = self.get_upstream_nodes(last_run, remove_duplicates=True)
        self.source = self.create_graph(
            self.dag_edges, img_format=self.img_format, view=self.view
        )
        return self.source

    @lru_cache()
    @beartype
    def lineage_to_edges(self, lineage_string: str) -> list[Tuple]:
        """
        convert the get_artifact_from_lineage lineage string into nodes
        e.g. A->B->C to [(A,B), (B,C)]
        Args:
            lineage_string (str): [description]

        Returns:
            list[Tuple]: [description]
        """
        lineage = []
        prev_run = None
        for run in lineage_string.split("->"):
            if prev_run is not None:
                lineage.append((prev_run, run))
            prev_run = run
        return lineage

    @lru_cache()
    @beartype
    def get_upstream_nodes(self, run: str, remove_duplicates: bool = True) -> List:
        """
        Iterates into the linked runs and returns the linkages.
        """

        def _get_srcrun_connections(run: str, upstream_runs: List = None) -> List:
            if upstream_runs is None:
                upstream_runs = []
            run_obj = _process_run(run)
            last_task = get_last_finished_task(run_obj)
            if (
                data := last_task.data
            ):  # using last finished step as artifacts accumulate in metaflow
                if new_artifact := ArtifactMatchers.prefix(
                    data, UPSTREAM_PIPELINE_LINKING_PREFIX
                ):
                    for _, artf_value in new_artifact.items():
                        if artf_value["value"] is not None:
                            upstream_runs.append((run, artf_value["value"]))
                            _get_srcrun_connections(artf_value["value"], upstream_runs)
                return upstream_runs

        connections = _get_srcrun_connections(run)
        if remove_duplicates:
            connections = list(set(connections))

        return connections

    @beartype
    def create_graph(self, dag_edges: list[Tuple], **kwargs):
        self.parse_kwargs(**kwargs)
        dag_edges_set = sorted(sum([[x[0], x[1]] for x in dag_edges], []))
        num_dag_edges = len(dag_edges_set)

        import string

        letters = [
            x + y for x in string.ascii_uppercase for y in string.ascii_uppercase
        ]
        edge_letters = {
            edg: lt for lt, edg in zip(letters[: num_dag_edges + 1], dag_edges_set)
        }

        connections = [(edge_letters[x[0]], edge_letters[x[1]]) for x in dag_edges]

        self.dag = graphviz.Digraph(strict=True, format=self.img_format)
        for edg, ltr in edge_letters.items():
            self.dag.node(ltr, edg)
        self.dag.edges(connections)

        with tempfile.NamedTemporaryFile(suffix=f".{self.img_format}") as tf:
            fn = tf.name.split(".", maxsplit=1)[0]
            self.dag.render(fn, cleanup=True)
            self.dag.render(tf.name, view=self.view)
            self.bytes = base64.encodebytes(tf.read())
        return self.dag.source

    @beartype
    def dot_to_ascii(self, dot: str, fancy: bool = True) -> str:
        """
        Converts a dot source diagram into an ascii art representation.

        Args:
            dot (str): dot source representation (output of self.from_last_run or self.from_lineage)
            fancy (bool, optional): Use fancy boxes. Defaults to True.

        Returns:
            str: ascii art representation of the dag
        """

        url = "https://dot-to-ascii.ggerganov.com/dot-to-ascii.php"
        boxart = 0

        # use nice box drawing char instead of + , | , -
        if fancy:
            boxart = 1

        params = {
            "boxart": boxart,
            "src": dot,
        }

        response = requests.get(url, params=params).text

        if response == "":
            response = "DOT string is not formatted correctly"

        return response

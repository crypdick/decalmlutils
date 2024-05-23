import logging

import pandas as pd
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple, Union
from metaflow import Flow, Run, namespace
from metaflow.client.core import MetaflowData

from decalmlutils.mflow.flows import UPSTREAM_PIPELINE_LINKING_PREFIX
from decalmlutils.mflow.runs import _process_run, get_run_pathspec
from decalmlutils.mflow.tasks import get_last_finished_task
from decalmlutils.misc import flatten_list

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports


class DataArtifactNotFoundError(Exception):
    pass


@beartype
def get_artifact_from_run(
    run: Union[str, Run],
    artifact_name: str,
    legacy_names: Optional[list[str]] = None,
    missing_ok: bool = False,
) -> Any:
    """
    Given a MetaFlow Run and a key, scans the run's tasks and returns the artifact with that key.

    NOTE: use get_artifact_s3url_from_run() if you want the S3 URL for the artifact, not the artifact itself.

    This allows us to find data artifacts even in flows that did not finish. If we change the name of an artifact,
    we can support backwards compatibility by also passing in the legacy keys. Note: we can avoid this by resuming a
    specific run and adding a node which re-maps the artifact to another key. This will assign the run a new ID.

    Args:
        missing_ok: whether to allow an artifact to be missing
        artifact_name: name of the attribute to look for in task.data
        run: a metaflow.Run() object, or a run pathspec
        legacy_names: backup names to check

    Returns:
       the value of the attribute. If attribute is not found

    Raises:
        DataartifactNotFoundError if artifact is not found and missing_ok=False
        ValueError if Flow not found
        ValueError if Flow is found but run pathspec is not.
    """
    run = _process_run(run)
    legacy_names = legacy_names or list()

    names_to_check = [artifact_name] + legacy_names

    # using last finished step as artifacts accumulate in metaflow
    logger.info(f"Fetching artifact `{names_to_check}` from {run}")
    last_task = get_last_finished_task(run, no_steps="raise")

    for name_ in names_to_check:
        if last_task.data is not None and name_ in last_task.data:
            return getattr(last_task.data, name_)

    if not missing_ok:
        raise DataArtifactNotFoundError(
            f"No data artifact with name {artifact_name} found in {run}. Also checked legacy names: {legacy_names}. "
            f"Available artifacts: {list(vars(last_task.data)['_artifacts'].keys())}"
        )


@beartype
def get_many_artifacts_from_run(
    run: Union[str, Run],
    artifact_names: list[Union[str, list[str]]],
    missing_ok: bool = False,
    legacy_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    An optimized version of get_artifact_from_run() for multiple artifacts.

    Args:
        missing_ok: whether to allow an artifact to be missing. If False, will check that
        artifact_names: names of the artifacts to look for. If a list, the first element is the primary name, and the other elements are legacy names.
        run: a metaflow.Run() object, or a run pathspec


    Returns:
       the value of the attribute. If attribute is not found

    Raises:
        DataartifactNotFoundError if artifact is not found and missing_ok=False
        ValueError if Flow not found
        ValueError if Flow is found but run pathspec is not.
    """
    legacy_names = legacy_names or list()
    names_to_check = artifact_names + legacy_names
    run = _process_run(run)

    artifacts = {}
    all_names = flatten_list(names_to_check)

    # using last finished step as artifacts accumulate in metaflow
    # note: do not use spinner() here bc it causes serialization errors
    last_task = get_last_finished_task(run)
    for name in all_names:
        if last_task.data is not None and name in last_task.data:
            artifacts[name] = getattr(last_task.data, name)

    if not missing_ok:
        names_found = artifacts.keys()
        missing_artifacts = []
        for item in artifact_names:
            if isinstance(item, str):
                if item not in names_found:
                    missing_artifacts.append(item)
            elif isinstance(item, list):
                found = [name for name in item if name in names_found]
                if len(found) == 0:  # if none of the names in the list were found
                    missing_artifacts.append(item)

            if missing_artifacts:
                if legacy_names:
                    logger.warning(
                        f"Artifacts not found in run {run}: {missing_artifacts}. Some of these may have been found under a different name: {names_found}"
                    )
                else:
                    raise DataArtifactNotFoundError(
                        f"Artifacts not found in run {run}: {missing_artifacts}."
                    )

    return artifacts


@beartype
def get_artifact_s3url_from_run(
    run: Union[str, Run],
    name: str,
    legacy_names: Optional[list[str]] = None,
    missing_ok: bool = False,
) -> str:
    """
    Given a MetaFlow Run and a key, scans the run's tasks and returns the artifact's S3 URL with that key.

    NOTE: use get_artifact_from_run() if you want the artifact itself, not the S3 URL to the artifact.

    This allows us to find data artifacts even in flows that did not finish. If we change the name of an artifact,
    we can support backwards compatibility by also passing in the legacy keys. Note: we can avoid this by resuming a
    specific run and adding a node which re-maps the artifact to another key. This will assign the run a new ID.

    Args:
        missing_ok: whether to allow an artifact to be missing
        name: name of the attribute to look for in task.data
        run: a metaflow.Run() object, or a run pathspec
        legacy_names: backup names to check

    Returns:
       the value of the attribute. If attribute is not found

    Raises:
        DataartifactNotFoundError if artifact is not found and missing_ok=False
        ValueError if Flow not found
        ValueError if Flow is found but run pathspec is not.
    """
    run_obj = _process_run(run)
    legacy_names = legacy_names or list()
    names_to_check = [name] + legacy_names
    logger.info(f"Fetching artifact `{names_to_check}` from {run_obj}")

    # using last finished step as artifacts accumulate in metaflow
    last_task = get_last_finished_task(run_obj)

    for name_ in names_to_check:
        try:
            # checking if name_ in task.artifacts does not work
            return getattr(last_task.artifacts, name_)._object["location"]
        except AttributeError:
            pass

    if not missing_ok:
        raise DataArtifactNotFoundError(
            f"No data artifact with name {name} found in {run_obj}. Also checked legacy names: {legacy_names}"
            f"Available artifacts: {list(vars(last_task.data)['_artifacts'].keys())}"
        )


class ArtifactMatchers:
    """
    Functions which check whether artifacts in `data_list` fit the criteria.

    Signature:
    map(data_list, matching_key) --> list[<artifact values>]
    """

    @beartype
    def exact(data_list: MetaflowData, key: str) -> Dict:
        if key in data_list:
            return {key: {"value": getattr(data_list, key)}}
        else:
            return {}

    @beartype
    def prefix(data_list: MetaflowData, prefix: str) -> Dict:
        matches = [itm for itm in data_list._artifacts if itm.startswith(prefix)]
        return {mch: {"value": getattr(data_list, mch)} for mch in matches}


@beartype
def get_artifact_from_lineage_no_metadata(
    key: str, legacy_keys: Optional[list[str]] = None, **kwargs
) -> Any:
    """
    Abstraction for get_artifact_from_lineage() which just gives back the artifact.

    To search a specific lineage, use the `run` kwarg. Otherwise, it will search the lineage of the
    current run.

    Args:
        key: key of artifact. Must be an exact match, and must be unique in the lineage.
        legacy_keys: other keys to search for with artifact

    Returns:
        the artifact


    Raises:
        See get_artifact_from_lineage()
    """
    lineage_being_searched = kwargs.get("run", "current lineage")
    # note: tried using spinner() here but got NameError: name '_last_cols' is not defined
    logger.info(f"Fetching artifact `{key}` from {lineage_being_searched}")
    missing_ok = kwargs.pop("missing_ok", False)
    _lineage, _artifacts = get_artifact_from_lineage(
        key, legacy_keys=legacy_keys, missing_ok=missing_ok, **kwargs
    )

    if not missing_ok:
        # FIXME this does not raise if the key exists and the value is None
        assert (
            len(_artifacts) == 1
        ), f"Expected one artifact but found {len(_artifacts)}: {_artifacts} \n lineage searched: {_lineage}"

    if len(_artifacts) == 0 and missing_ok:
        artifact = None
    else:
        # everything is ok
        _artifact = _artifacts[0]
        key_found = list(_artifact)[0]
        artifact = _artifact[key_found]["value"]

    return artifact


@beartype
def get_artifact_from_lineage(
    key: str,
    run: Union[str, Run, None] = None,
    depth: int = -1,
    missing_ok: bool = True,
    legacy_keys: Optional[list[str]] = None,
    pipeline_prefix=UPSTREAM_PIPELINE_LINKING_PREFIX,
    match: str = "exact",
    checked_srcruns: Optional[set] = None,
    specific_flow: Union[str, Run, None] = None,
    specific_run: Union[str, Run, None] = None,
    parent_run: Optional[str] = None,
) -> Tuple[set[str], list[Dict[str, Dict[str, Any]]]]:
    """
    "Searches within the pipeline lineage for artifacts matching the `key` (and legacy_keys)

    Example usage:
    lineage, artifacts = get_artifact_from_lineage("first_item", "fourth/1628868532342926")

    lineage = set(["first/1628868225323477", "fourth/1628868532342926", "second/1628868232938746", "third/1628868241168329"])

    artifacts = [
        dict(
            first_item = dict(
                value = "first-b",
                lineage = "fourth/1628868532342926->third/1628868241168329->second/1628868232938746->first/1628868225323477",
                flow = "first",
                run = "first/1628868225323477")
    ]

    IMPORTANT:
    While searching the lineage, it will search all pipelines at each depth and collect the first
    item it finds in each of the pipeline. If multiple items are retrieved i.e. more than 1 pipeline contains
    the artifact, all of them are returned. If you want an artifact with ambiguous name, pass the specific_flow or
    specific_run to disambiguate.

    LIMITATION:
    If we set `srcrun_foo` in a `start` step, we cannot use this utility yet! We must wait for `start` to finish so that
    the `self.srcrun_foo` artifact is saved the datastore. /Then/ we can use the MetaFlow client API to look up
    artifacts which start with pipeline_prefix

    Args:
        key: name of artifact (or prefix)
        run: run_id or metefalow run object which we use to start searching
        depth: number of upstream pipelines to search, defaults to -1 which means search till infinity
        missing_ok: if artifact not found, throw error or return empty list, defaults to True
        legacy_keys: alternate keys, defaults to None
        pipeline_prefix: prefix to search for upstream pipelines, defaults to "srcrun_"
        match: criteria to match artifact names by: "exact" or "prefix", defaults to "exact"
        checked_srcruns: set containing the pipelines which have been searched, defaults to None
        specific_flow: name of flow if there is a specific flow from which artifact needs to be found, defaults to None
        specific_run: name of run if there is a specific run from which artifact needs to be found, defaults to None
    Returns:
        tuple containg (a) set of pipelines which were search, (b) List of dicts. Each dict corresponds to an artifact, where the key is the
        artifact name and value is the artifact info. Artifact info is a dict with keys `lineage` and `value`(of artifact).
    Raises:
        ValueError: if the run provided is not a valid string representation of a Run
        ValueError: if specifed_run is neither a valid run pathspec nor a string
        ValueError: if specifed_flow is neither a valid flow pathspec nor a string
        DataArtifactNotFoundError: if missing_ok is set to False and no artifact matching the key(s) is found
    """

    if match not in ["exact", "prefix"]:
        raise KeyError(
            f"Matching function not found. Please provide either 'exact' or 'prefix'. You provided: {match}"
        )

    namespace(None)
    match_func = getattr(ArtifactMatchers, match)

    # init flow, run_obj, run vars
    run = run or get_run_pathspec()
    run_obj = _process_run(run)
    run = run_obj.pathspec
    flow = str(run.split(sep="/")[0])

    if specific_flow is not None:
        if isinstance(specific_flow, str):
            specific_flow = specific_flow.split("/")[0]
        elif isinstance(specific_flow, Flow):
            specific_flow = specific_flow.pathspec
        else:
            raise ValueError(
                "Unable to process valid pathspec from specific_flow. Please provide the flow name or flow_obj."
            )

    if specific_run is not None:
        if isinstance(specific_run, str):
            specific_run = f"{flow}/{specific_run.split('/')[-1]}"
        elif isinstance(specific_run, Run):
            specific_run = specific_run.pathspec
        else:
            raise ValueError(
                "Unable to process valid pathspec from specific_run. Please provide the flow name or flow_obj."
            )

    lineage = run if parent_run is None else f"{parent_run}->{run}"

    checked_srcruns = checked_srcruns or set()
    if run in checked_srcruns:
        return checked_srcruns, []
    else:
        checked_srcruns.add(run)

    artifacts = list()
    upstream_runs = dict()
    keys_to_check = [key] + (legacy_keys or [])

    # NOTE: now using a helper to get the last finished step
    last_task = get_last_finished_task(run_obj, no_steps="raise")
    if (
        data := last_task.data
    ):  # using last finished step as artifacts accumulate in metaflow
        if specific_flow in [flow, None] and specific_run in [run, None]:
            for check_key in keys_to_check:
                if new_artifact := match_func(data, check_key):
                    # add lineage to each artifact dict
                    for artf_name in new_artifact:
                        new_artifact[artf_name].update(
                            {
                                "lineage": lineage,
                                "flow": flow,
                                "run": run,
                            }
                        )
                    new_artifact_list = multikey_dict_to_singlekey_dict_list(
                        new_artifact
                    )
                    artifacts.extend(new_artifact_list)
                    return checked_srcruns, artifacts
        upstream_runs.update(ArtifactMatchers.prefix(data, pipeline_prefix))

    if depth != 0:
        for _, srcrun in upstream_runs.items():
            # This is to handle the edge case where we use Nonetype values with srcrun_ prefixed variables,
            # e.g. Task('InferencePipeline/2588/start/22026').data.srcrun_train2inference
            if srcrun is None or srcrun["value"] is None:
                continue
            else:
                upstream_run = srcrun["value"]
                # metadata for some bce-internal layers (Ex: 5688) contain the string 'manual'
                # where there should be an MLInternalLayerPipeline pathspec. This is because this metadata
                # was backfilled manually. This causes the JiraHintRasterPipeline to fail
                # (https://biocarbon.atlassian.net/browse/SML-1027)
                # we should back off from looking at these runs anyway, since they're a dead end
                # and the high prediction frequency alerts will still be found if we just consider this a dead end.
                if upstream_run == "manual":
                    continue
            deeper_checked_runs, deeper_artifacts = get_artifact_from_lineage(
                key=key,
                run=upstream_run,
                depth=depth - 1,  # reduce depth by 1 every recursion
                missing_ok=True,
                legacy_keys=legacy_keys,
                match=match,
                checked_srcruns=checked_srcruns,
                parent_run=lineage,
                pipeline_prefix=pipeline_prefix,
                specific_flow=specific_flow,
                specific_run=specific_run,
            )
            checked_srcruns.update(deeper_checked_runs)
            artifacts.extend(
                artf
                for artf in deeper_artifacts
                if _artifact_not_in_list(artf, artifacts)
            )

    if not artifacts and not missing_ok:
        raise DataArtifactNotFoundError(
            f"No data artifact that matches `{key}` found in `{run_obj}`. Also checked legacy names: {legacy_keys}. "
            f"Runs checked: {checked_srcruns}"
        )

    return checked_srcruns, artifacts


@beartype
def _artifact_not_in_list(artf: Dict, artifacts: List) -> bool:
    """
    _artifact_not_in_list checks if artf is a member of artifacts and handles ValueErrors for dataframes.

    Args:
        artf (Dict): artifact to check
        artifacts (List): list of artifacts to check against

    Returns:
        bool: True if not a member or incomparible, else false
    """
    # simple - works for most types
    try:
        return artf not in artifacts

    # for all other types that raise ValueError on comparison e.g. dataframes
    except ValueError:
        artf_name = list(artf.keys())[0]
        artf_value = artf[artf_name]["value"]
        artf_value_type = type(artf_value)

        same_name_artifacts = [x for x in artifacts if artf_name in x.keys()]
        same_type_artifact_values = [
            x[artf_name]["value"]
            for x in same_name_artifacts
            if type(x[artf_name]["value"]) == artf_value_type
        ]

        not_in = True
        for x in same_type_artifact_values:
            try:
                not_in = x != artf_value
                if isinstance(not_in, (pd.DataFrame, pd.Series)):
                    not_in = not_in.any(None)
                if not (not_in):
                    return not_in
            except ValueError:
                continue
        return not_in


@beartype
def multikey_dict_to_singlekey_dict_list(dictionary: Dict) -> list[Dict]:
    """
    Converts a mutlikey dict like {"a": 1, "b": 2} into a singlekey dict list like [ {"a": 1} , {"b": 2}]

    Args:
        dictionary (Dict): mutlikey dictionary

    Returns:
        list[Dict]: list of singlekey dictionaries
    """
    list_of_dicts = list()
    for k, v in dictionary.items():
        list_of_dicts.append({k: v})
    return list_of_dicts

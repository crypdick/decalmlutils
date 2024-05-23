import datetime
import logging
import operator

import pandas as pd
from beartype import beartype
from beartype.typing import Callable, Dict, List, Optional, Tuple, Union
from metaflow import Flow, Metaflow, Run, current, namespace
from metaflow.exception import MetaflowNotFound

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports


class RunspecNotFoundError(Exception):
    pass


@beartype
def get_latest_successful_run(
    flow_name: Union[str, Flow],
) -> Optional[Run]:
    """
    Get the latest successful run of a flow.

    Args:
        flow_name: the name of the flow

    Returns:
        The latest successful run of the flow, or None if no successful runs are found.
    """
    namespace(None)
    return Flow(flow_name).latest_successful_run


@beartype
def get_run_pathspec() -> str:
    """
    Retrieve the Run pathspec of the currently executing Metaflow job.

    Fails very fast if called outside a Metaflow run

    Raises:
        RunspecNotFoundError: If the runspec cannot be found

    Returns:
        str: The runspec of the currently executing Metaflow job
    """
    try:
        run_pathspec = Flow(current.flow_name)[current.run_id].pathspec
    except TypeError as e:
        raise RunspecNotFoundError(
            # NOTE: if this msg is updated, please update the exception catching in update_ml_layer_metadata
            "MetaFlow failed to find current Flow. `get_run_pathspec()` can only be used within the context of "
            "MetaFlow runs."
        ) from e

    return str(run_pathspec)


@beartype
def run_finished(run: Union[str, Run]) -> bool:
    """
    Returns True if the run has finished, False otherwise.
    """
    run_obj = _process_run(
        run
    )  # do not overwrite `run` bc it can have side-effects upstream
    return run_obj.finished


@beartype
def get_runs_with_tags(
    flow_name: str,
    tags: Optional[list[str]] = None,
    run_filter_func: Optional[Callable] = None,
    namespace_override: Optional[str] = None,
    successful_only: bool = True,
    min_run_id: Optional[int] = 0,
    max_run_id: Optional[int] = int(1e16),
) -> list[Run]:
    """
    Given a Flow name and a list of tags, returns all run pathspecs that match.

    Args:
        flow_name: the name of the Flow to search
        tags: a list of tags to filter by. If None, all runs of the given flow will be returned.
        run_filter_func: a function that takes a Run object and returns True if the run should be included in the results.
        namespace_override: if provided, the namespace will be switched to this value for the duration of this function call.
        successful_only: if True, only successful runs will be returned.
        min_run_id: the miniumum run id of the flow after which to stop scanning. Defaults to 0 i.e. scan all runs.
        max_run_id: the maximum run id of the flow above which it will skip the run. Defaults to 1e8 i.e. scan all runs.

    Returns:
        list[Run]: list of Run objects that match the tags and run_filter_fun

    Side Effects:
        if `namespace_override` is passed, MetaFlow will switch namespaces. This namespace will persist outside this
        function call.
    """
    tags = tags or []
    run_filter_func = run_filter_func or (lambda x: True)
    namespace(namespace_override)

    matching_runs = []
    for run in Flow(flow_name).runs(
        *tags
    ):  # this will per-filter the loop based on tags, significantly speed-ups
        if int(run.id) < min_run_id:
            break  # went deep enough
        elif int(run.id) > max_run_id:
            continue  # go deeper
        elif run_filter_func(run):
            if successful_only and not run.successful:
                continue
            matching_runs.append(run)

    return matching_runs


@beartype
def _get_previous_runs(
    flow_name: str,
    attributes_config: dict = None,
    new_flow_col: Optional[str] = None,
    namespace_override: Optional[str] = None,
    tags: Optional[List] = None,
    run_filter_func: Optional[Callable] = None,
    min_run_id: Optional[int] = 0,
    max_run_id: Optional[int] = int(1e16),
) -> pd.DataFrame:
    """
    Get Previous Runs. Prefer using `get_runs_with_tags` instead.

    Goes through the metaflow artifact store and finds all /successful/ runs of the flow_name provided.
    To retrieve some attribute from the runs, provide a attributes_config in the following format:
        {
            attribute_name: {
                "if_missing": <value to assume if attribute can't be found>
                "legacy": <older name of attribute which will be tried if main name fails>
            }
        }
    NOTE 1: attribute_config is optional, if None is provided, no attributes will be returned, instead only pathspecs will be in the result e.g.
    NOTE 2: in the attribute_config, you can provide an empty dict or None for each attribute_name if no if_missing or legacy criteria is needed e.g.

        {
            attribute_name1: None,
            attribute_name2: {}
        }
    NOTE 3: created_at and finished_at columns are included by default

    Args:
        flow_name (str): Full name of the Flow
        attributes_config (dict, optional): config dict for the attributes to find. Defaults to None.
        new_flow_col (str, optional): Name of the new column with the pathspec. Defaults to flow_name
        namespace_override (str, optional): use a different namespace. Defaults to None. NOTE: this namespace will persist beyond this function
        tags (Optional[list[str]], optional): tags to filter the runs by
        run_filter_func (Optional[Callable], optional): if provided, the run object will be passed to this function and only included if it returns true
        min_run_id (Optional[int], optional): the miniumum run id of the flow after which to stop scanning. Defaults to 0 i.e. scan all runs.
        max_run_id (Optional[int], optional): the maximum run id of the flow above which it will skip the run. Defaults to 1e8 i.e. scan all runs.

    Returns:
        pd.DataFrame: df with a column for the flow's pathspec and a column for each attribute in the attributes_config

    Side Effects:
        if `namespace_override` is passed, MetaFlow will switch namespaces. This namespace will persist outside this
        function call.
    """
    matching_runs = get_runs_with_tags(
        flow_name=flow_name,
        tags=tags,
        run_filter_func=run_filter_func,
        namespace_override=namespace_override,
        min_run_id=min_run_id,
        max_run_id=max_run_id,
        successful_only=True,
    )
    attributes_config = dict(attributes_config or {})
    attributes_config.pop("finished_at", None)
    attributes_config.pop("created_at", None)

    flow_col = new_flow_col or flow_name
    all_run_attributes = []
    for run in matching_runs:
        data = run.data
        run_attributes = {
            flow_col: run.pathspec,
            # https://dendrasystems.slack.com/archives/C02116BBNTU/p1646287103092299
            "finished_at": run.finished_at.astimezone(
                tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_at": run.created_at.astimezone(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }

        # if attribute_config is passed, populate them.
        for prm, conds in attributes_config.items():
            conds = conds or {}
            try:
                run_attributes[prm] = getattr(data, prm)
            except BaseException:
                if (alt := conds.get("legacy")) is not None:
                    try:
                        alternate = getattr(data, alt)
                    except BaseException:
                        alternate = conds.get("if_missing")
                else:
                    alternate = conds.get("if_missing")
                run_attributes[prm] = alternate

        all_run_attributes.append(run_attributes)

    # format all the info as a df
    all_run_attributes = pd.DataFrame(
        all_run_attributes,
        columns=[flow_col, "finished_at", "created_at"] + list(attributes_config),
    )

    return all_run_attributes


@beartype
def filter_runs_by_attributes(
    flow_name: str,
    attributes_config: Dict,
    new_flow_col: Optional[str] = None,
    namespace_override: str = None,
    tags: Optional[List] = None,
    run_filter_func: Optional[Callable] = None,
    min_run_id: Optional[int] = 0,
    max_run_id: Optional[int] = int(1e16),
    allowed_modes: Optional[Tuple[str, ...]] = ("prod",),
) -> pd.DataFrame:
    """
    Filter Runs by Attributes.

    Wrapper around _get_previous_runs which also filters out the runs based on the attributes_config. For this,
    the attributes_config needs the same attributes_config as the _get_previous_runs but 2 more fields called
    `match` and `operand` for each of the attribute on which filtering needs to be done, e.g.

        {
            "threshold": {
                "match": 0.8,
                "operand": "eq",
                "if_missing": {
                    "value": <value>,
                    "eval": callable,
                    "legacy": <legacy_attribute_name>
                    },
            }
        }

        threshold is the attribute name
        match (Optional): this value is matched. NOTE: if the value is a list, the df[col].isin() pattern is used to cover both [1,2] and ['1','2']
                    as both are same for JSONType inputs like impossible_classes, class_to_skip, etc.)
        operand (Optional): defaults to eq and determines what operand to use against the match value. For `isin`, the values need to be a list
        eval (Optiona;): if true, the value of this attribute will be eval-ed - this is useful for attributes which changed from str to JSONType over time
        if_missing (Optional): if the attribute is not found, this value will be imputed - useful if new attributes were added over time

    For attributes that do not have "match", they will be returned in the final df but no filtering on their value will be done

    NOTE: if match values are JSONType e.g. list or dicts, use isin and provide all possible variations e.g.
        {
            "impossible_classes": {
                "match": [['506'], [506], "['506']", '["506"]'],
                "operand": "isin"
                }
        }

    Args:
        flow_name (str): Full name of the Flow
        attributes_config (dict, optional): config dict for the attributes to find. Defaults to None.
        new_flow_col (str, optional): name for the new column. Defaults to the flow_name
        namespace_override (str, optional): use a different namespace. Defaults to None. NOTE: this namespace will persist beyond this function
        tags (Optional[list[str]], optional): tags to filter the runs by
        run_filter_func (Optional[Callable], optional): if provided, the run object will be passed to this function and only included if it returns true
        min_run_id (Optional[int], optional): the miniumum run id of the flow after which to stop scanning. Defaults to 0 i.e. scan all runs.
        max_run_id (Optional[int], optional): the maximum run id of the flow above which it will skip the run. Defaults to 1e8 i.e. scan all runs.
        allowed_modes (Optional[Tuple], optional): if provided, attributed_config will include "mode": {"match": allowed_modes, "operand": "isin"}, to
            filter out runs not in the allowed_modes. if None, no filtering is done and all modes included. Default: ("prod",)

    Returns:
        pd.DataFrame: df with a column for the flow's pathspec and a column for each attribute in the attributes_config
    """
    if allowed_modes is not None:
        logger.info(f"Filtering runs allowing only these modes: {allowed_modes}")
        attributes_config["mode"] = {"match": allowed_modes, "operand": "isin"}
    else:
        logger.warning(
            "Allowed modes is `None`. All modes (prod, test, smoke_test, smoke_test, etc) will be included in the result."
        )

    runs_attribute_df = _get_previous_runs(
        flow_name=flow_name,
        new_flow_col=new_flow_col,
        namespace_override=namespace_override,
        attributes_config=attributes_config,
        tags=tags,
        run_filter_func=run_filter_func,
        min_run_id=min_run_id,
        max_run_id=max_run_id,
    )

    def add_condition(conditions, new_condition):
        if conditions is None:
            return new_condition
        else:
            return conditions & new_condition

    if runs_attribute_df.size == 0:
        pass
    else:
        conditions = None
        for attr, conf in attributes_config.items():
            if conf is None or (match_crit := conf.get("match")) is None:
                continue
            if (operand := conf.get("operand", "eq")) in [
                "ge",
                "gt",
                "le",
                "lt",
                "eq",
                "ne",
            ]:
                operand = getattr(operator, operand)
                conditions = add_condition(
                    conditions, operand(runs_attribute_df[attr], match_crit)
                )
            elif operand == "isin":
                conditions = add_condition(
                    conditions, runs_attribute_df[attr].isin(match_crit)
                )
            else:
                raise ValueError(f"Operand `{operand}` not supported!.")

        if conditions is not None:
            runs_attribute_df = runs_attribute_df[conditions].reset_index(drop=True)
    return runs_attribute_df


@beartype
def _process_run(run: Union[str, Run]) -> Run:
    """
    Checks if run is a string or a Run object and returns the Run object.

    Raises:
        ValueError: if Run pathspec does not exist
        ValueError: If parent Flow does not exist
    """
    namespace(None)  # allows us to access all runs in all namespaces
    if isinstance(run, str):  # convert to Run object
        try:
            run = Run(run)
        except Exception as e:
            # run pathspec not found. see if we can find other runs and list them
            flow = run.split(sep="/")[0]
            try:
                flow = Flow(flow)
                raise ValueError(
                    f"Could not find run pathspec `{run}`. Possible values: {list(flow.runs())}"
                ) from e
            except MetaflowNotFound as e2:
                if len(Metaflow().flows) < 2:
                    raise ValueError(
                        f"Could not find flow {flow}. Only found {len(Metaflow().flows)} flows, which"
                        f"suggests that you forgot to setup MetaFlow credentials"
                    ) from e2
                else:
                    raise ValueError(
                        f"Could not find flow `{flow}`. This can happen if you didn't set up Metaflow "
                        f"credentials. Available flows: {Metaflow().flows}"
                    ) from e2
    return run

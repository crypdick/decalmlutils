import copy
import datetime
import logging
import os
import time
from decimal import Decimal
from logging import getLogger

import botocore.exceptions
from beartype import beartype
from beartype.typing import Dict, Iterable, Optional, Union
from metaflow import Flow, current
from tenacity import retry, stop_after_attempt, wait_random_exponential

from decalmlutils.conf import settings
from decalmlutils.io.aws import get_aws_client
from decalmlutils.io.aws.sns import publish_to_sns
from decalmlutils.git_.git_ import get_current_local_branch, get_current_local_commit
from decalmlutils.slack.slack import METRICS_ERRORS_CHANNEL, Slacker
from decalmlutils.mflow.artifacts import get_artifact_from_lineage
from decalmlutils.mflow.cloudwatch_logs import get_cloudwatch_log_link
from decalmlutils.mflow.flows import UPSTREAM_PIPELINE_LINKING_PREFIX
from decalmlutils.misc import millify
from decalmlutils.ds.tensors import get_chunks


def log_metrics_to_cloudwatch(
    scalar_metrics,
    distribution_metrics,
    pipeline_runtime_sec: float,
    is_prod: bool = False,
) -> None:
    """
    Logs scalar metrics and distribution metrics to Cloudwatch Metrics.

    Args:
        scalar_metrics: stand-alone metrics
        distribution_metrics: collections of related metrics
        pipeline_runtime_sec: num secs it took to run the pipeline
        is_prod: whether this is called from prod mode. If False, will log to a `ML-PYTEST` namespace so that it does
            not pollute the `ML` namespace.
    """
    logger = logging.getLogger(__name__)
    """
    Setup default vals.
    """
    current_flow = current.flow_name
    assert current_flow is not None, "Error retrieving flow name"
    default_dimensions = [{"Name": "Flow", "Value": current_flow}]
    """
    Manually construct metric for elapsed time.
    """
    metrics_formatted = [
        {
            "MetricName": f"{current_flow} runtime",
            "Value": pipeline_runtime_sec,
            "Unit": "Seconds",
            "Dimensions": default_dimensions,
        }
    ]
    """
    Scalar metrics.
    """
    if scalar_metrics is not None:
        for metric_name, metric_value in scalar_metrics.items():
            assert isinstance(
                metric_value, (int, float, Decimal)
            ), f"Scalar metric value must be numeric, got {metric_value} type {type(metric_value)} for {metric_name}"
            assert isinstance(
                metric_name, str
            ), f"Scalar metric name must be a string, got {metric_name} type {type(metric_name)}"
            metrics_formatted.append(
                {
                    "MetricName": metric_name,
                    "Value": metric_value,
                    "Unit": "None",
                    "Dimensions": default_dimensions,
                }
            )

    """metric collections"""
    if distribution_metrics is not None:
        for metric_collection_name, metric_payload in distribution_metrics.items():
            assert isinstance(metric_collection_name, str), (
                f"Distribution metric collection name must be a string. Got {metric_collection_name} "
                f"type {type(metric_collection_name)}"
            )
            for metric_name, metric_value in metric_payload.items():
                assert isinstance(metric_value, (int, float, Decimal)), (
                    f"invalid metric value {metric_value}, type {type(metric_value)} for {metric_name}"
                    f"for metric {metric_name} (allowed types: int, float, Decimal)"
                )
                dims_this_metric = copy.deepcopy(default_dimensions)
                # each metric collection will have a separate section in the CloudWatch hierarchy, but each metric
                # won't. this separation is needed for Cloudwatch alarms because alarms can only be set on single
                # timeseries.
                # the '-' value is needed because otherwise we fail boto's param validation. however, this leading
                # hyphen is deleted by cloudwatch dashboards.
                dims_this_metric.append({"Name": metric_collection_name, "Value": "-"})
                for dims in dims_this_metric:
                    assert isinstance(
                        dims["Value"], str
                    ), f'Dimension value must be a string. Got {dims["Value"]} type {type(dims["Value"])}'

                dimension_metric = {
                    "MetricName": metric_name,
                    "Value": metric_value,
                    "Unit": "None",  # make all metrics dimensionless instead of having to specify each one separately
                    "Dimensions": dims_this_metric,
                }

                metrics_formatted.append(dimension_metric)
    logger.info("Formatted metrics")
    """
    post.
    """
    post_metrics_to_cloudwatch(metrics_formatted, is_prod=is_prod)


# retry needed due to occasionally flakey handshake in CI
@retry(
    wait=wait_random_exponential(multiplier=0.1, max=0.2),
    reraise=True,
    stop=(stop_after_attempt(10)),
)
@beartype
def post_metrics_to_cloudwatch(metrics: list[Dict], is_prod: bool) -> None:
    """
    Send custom metrics, 20 at a time.

    if this is called outside of prod mode, it'll post to a separate namespace. this is useful for making sure that the
    metrics param is well-formed.

    Note: the log_metrics_to_cloudwatch function should be preferred over this function. for some custom
    multi-dimensional metrics, we need to format the metrics manually and call this function.

    metrics: list of dict, where each dict looks like:
                    {'MetricName': 'F1', Value: 0.9, Unit: 'None', Dimensions: {Layer: layer_id, Class: Gamba Grass, ...}}
    """
    logger = getLogger(
        os.path.basename(__file__)
    )  # no cloudwatch logging -> this func can be used outside of metaflow
    client = get_aws_client("cloudwatch")
    MAX_METRICS_PER_CALL = 20  # this limit is imposed by Cloudwatch
    chunks = get_chunks(num_items=len(metrics), chunk_size=MAX_METRICS_PER_CALL)
    for chunk_start, chunk_end in chunks:
        chunk = metrics[chunk_start:chunk_end]
        namespace = (
            "ML" if is_prod else "ML-PYTEST"
        )  # do not pollute ML namespace during tests
        try:
            client.put_metric_data(Namespace=namespace, MetricData=chunk)
        except botocore.exceptions.ParamValidationError as e:
            logger.info(f"Cloudwatch did not like this parameter chunk: {chunk}")
            raise e
        except BaseException as e:
            logger.warning(f"unhandled error occurred: {e}")
            # do not raise e so that we don't fail the entire pipeline
            # send slack alerts when cloudwatch metrics logging fails
            publish_to_sns(
                f"Unhandled cloudwatch error: {e}",
                is_prod=is_prod,
                channel=METRICS_ERRORS_CHANNEL,
            )


@beartype
def alert_job_finished(
    mode: str,
    start_time: float,  # stop making this Optional!
    msg: str = "",
    channel: str = settings.ML_ALERTS_CHANNEL,
    scalar_metrics: Optional[Dict[str, Union[int, float]]] = None,
    distribution_metrics: Optional[Dict[str, Dict[str, Union[int, float]]]] = None,
    figures: Optional[Iterable[bytes]] = None,
) -> None:
    """
    Convenience function to log metrics and send alerts. Intended to be used in `end` step of Metaflow Flows.

    Note: this used to live in the SNS module. however, it was moved here to avoid circular dependencies.

    Args:
        msg: an arbitrary string to include in alerts. Mandatory arg.
        mode: what mode the Flow was run. If not `prod`, this function short circuits.
        start_time: time at which the flow was started. Ingests `self.start_time = time.time()` from the flow's `start`.
        scalar_metrics: A dictionary of numerical metrics. These get logged to both Cloudwatch Metrics and the alert.
        distribution_metrics: A dictionary of distribution metrics. Each distribution metric is also a dict which
            contains a collections of related metrics that are plotted in the same Cloudwatch plot. These are logged
            to Cloudwatch only for anomaly tracking (not included in alerts).

            For example:
            {'Collection name 1': {'metric 1': 1, 'metric 2': 2, 'metric 3': 3},
             'Collection name 2': {'metric 1': 1, 'metric 2': 2, 'metric 3': 3}}
    """
    logger = logging.getLogger(__name__)  # do not create in global scope
    is_prod = mode == "prod"
    if scalar_metrics is None:
        scalar_metrics = {}

    run_obj = Flow(current.flow_name)[current.run_id]
    run_pathspec = run_obj.pathspec
    metadata = [
        f":woohoo: Job: `{run_pathspec}`",
        f":eyebrows: Logs: {get_cloudwatch_log_link()}",
    ]

    sec_elapsed = round(time.time() - start_time)
    # sec to "7 days, 17:12:57" format
    runtime = f":clock1: Duration: `{str(datetime.timedelta(seconds=sec_elapsed))}`"
    metadata.append(runtime)
    # runtime automatically gets added to scalar metrics downstream

    tags = run_obj._tags
    alert_tags = [
        x
        for x in tags
        if not any(
            prefix in x
            for prefix in ["python_version:", "date:", "metaflow_version:", "runtime:"]
        )
    ]
    alert_tag_message = "`" + "` `".join(alert_tags) + "`"
    metadata.append(f":mirror: Tags: {alert_tag_message}")

    try:
        metadata.append(f":octocat: Branch: `{get_current_local_branch()}`")
        metadata.append(f":dusty_stick: Commit: `{get_current_local_commit(length=7)}`")
    except Exception as e:
        metadata.append(
            f":sad_dumpsterfire: Git Info: `Error` in getting git info: {e}."
        )

    try:  # prevent whole pipeline from failing if this block has trouble
        # get linked flows
        _, _artifacts = get_artifact_from_lineage(
            specific_run=run_pathspec,
            key=UPSTREAM_PIPELINE_LINKING_PREFIX,
            match="prefix",
        )
        linked_flows: list[str] = []
        for artifact in _artifacts:
            for artifact_name, payload in artifact.items():
                linked_pathspec = payload["value"]
                linked_flows.append(f"{artifact_name}=`{linked_pathspec}`")

        metadata.append(f":fistbump: Linked flows: {linked_flows}")
    except Exception:
        metadata.append(
            ":party_dumpsterfire: Linked flows: `Error` in getting linked flows."
        )

    """Finalize common metadata"""
    metadata.append(":grass:" * 10)
    """
    Log metrics.
    """
    # publish to cloudwatch
    logger.info(f"✍ Logging scalar metrics: {scalar_metrics}")
    logger.info(f"Logging distribution metrics: {distribution_metrics}")

    # pass cloudwatch logger so that we can get alerts if posting fails
    log_metrics_to_cloudwatch(
        scalar_metrics, distribution_metrics, sec_elapsed, is_prod=is_prod
    )
    # include in alert, too
    if scalar_metrics is not None:
        for k, v in scalar_metrics.items():
            millified_v = v
            if isinstance(millified_v, int):  # make large ints more human friendly
                try:  # millify() only works on positive values, does not handle nans or infs.
                    millified_v = millify(v, num_decimals=2)
                except Exception:
                    pass
            elif isinstance(
                millified_v, float
            ):  # make large floats more human friendly
                millified_v = f"{v:.2f}"
            metadata.append(f"{k}: `{millified_v}`")

    """Convert to string"""
    metadata.append(
        ""
    )  # finish metadata str with a newline, so that the `msg` starts on its own line
    metadata = "\n".join(metadata)

    # combine everything
    msg = "\n".join([metadata, msg])
    logger.info(f"✍ Logging alert: \n{msg}")

    slack = Slacker(is_prod=is_prod)
    slack.send_message(channel=channel, message=msg)

    if figures is not None:
        for fig in figures:
            slack.send_file(channel=channel, fig_bytes=fig)


def get_metrics_from_cloudwatch(
    metric_name: str, dimensions: list[Dict[str, str]], namespace: str = "ML"
):
    cloudwatch = get_aws_client("cloudwatch")

    # List metrics through the pagination interface
    paginator = cloudwatch.get_paginator("list_metrics")
    print(paginator.result_keys())

    for response in paginator.paginate(
        Dimensions=dimensions, MetricName=metric_name, Namespace=namespace
    ):
        print(response["Metrics"])


def get_all_ml_metric_keys():
    cloudwatch = get_aws_client("cloudwatch")

    # List metrics through the pagination interface
    paginator = cloudwatch.get_paginator("list_metrics")

    metric_keys = []

    for response in paginator.paginate(Namespace="ML"):
        metric_keys.extend(response["Metrics"])

    return metric_keys

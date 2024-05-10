import logging
import os
import sys
import warnings

from beartype.typing import Dict

logger = logging.getLogger(
    __name__
)  # do not use cloudwatch logging here due to circular imports


def assert_cli_args(**kwargs):
    """
    Decorate Metaflow Flows with this to assert certain arguments like --max-workers, etc.

    Usage:
        @assert_cli_args(**{"--max-workers": "1", "--max-num-splits": "100"})
        class MyFlow(FlowSpec): ...
    """

    def decorator(pipeline):
        def wrapper():
            if "run" not in sys.argv:  # so that this only executes when we run
                return pipeline()
            problems = []
            for arg, expected_value in kwargs.items():
                if arg in sys.argv:
                    ind = sys.argv.index(arg)
                    if (actual_value := sys.argv[ind + 1]) != expected_value:
                        problems.append(
                            {
                                "argument": arg,
                                "expected_value": expected_value,
                                "actual_value": actual_value,
                            }
                        )
            assert len(problems) == 0, f"CLI argument assertions failed: {problems}"
            logger.info("Asserting arguments passed!")
            return pipeline()

        return wrapper

    return decorator


def fix_cli_args(**kwargs: Dict[str, str]):
    """
    Decorator to override Metaflow CLI arguments.

    Usage:
        @fix_cli_args(**{"--max-workers": "1", "--max-num-splits": "100"})
        class InferencePipeline(FlowSpec): ...

    Warnings:
        If the argument is specified by the user, it will be overridden by the value specified in the decorator and a
        warning will be raised.
    """

    def decorator(pipeline):
        def wrapper():
            if "run" not in sys.argv and "resume" not in sys.argv:
                # ignore this decorator if we are not running or resuming a flow
                return pipeline()
            for arg, val in kwargs.items():
                if arg in sys.argv:  # if arg was passed, override it
                    ind = sys.argv.index(arg)
                    logger.warning(
                        f"`{arg}` arg was passed with value `{sys.argv[ind + 1]}`. However, this value will"
                        f"be overriden by @fix_cli_args with value {val}"
                    )
                    sys.argv[ind + 1] = val  # replace the val
                else:  # otherwise, add (arg, val) to the call
                    sys.argv.extend([arg, val])
            logger.info(f"Fixed CLI args for {kwargs.keys()}")
            return pipeline()

        return wrapper

    return decorator


def drop_cli_args(*arguments):
    """
    Decorate Metaflow Flows with this to drop certain arguments and their values like --max-workers, etc.

    Usage:

    @drop_cli_args("--max-workers", "--max-num-splits")
    class InferencePipeline(FlowSpec):
    ...
    """

    def decorator(pipeline):
        def wrapper():
            if "run" not in sys.argv:  # so that this only executes when we run
                return pipeline()
            for arg in arguments:
                if arg in sys.argv:
                    ind = sys.argv.index(arg)
                    sys.argv.pop(ind)  # drop the arg
                    sys.argv.pop(ind)  # drop the val
            return pipeline()

        return wrapper

    return decorator


def change_warnings(pipeline):
    """
    Decorator to raise warnings before running the pipeline.

    pipeline: the decorated pipeline
    """

    def give_warnings():
        if os.environ.get("MODE"):
            warnings.warn(
                """
            The environment variable, `MODE` is no longer used.
            Instead you must now set METAFLOW_PROFILE to:
            1. `prod`
                For production use cases and using S3 as metadata and data store.

                - To run `smoke-test` pass in a flag --smoke-test true

            2. `test`

                For pytest and using local disk as metadata and data store
            """
            )

        else:
            return pipeline()

    return give_warnings

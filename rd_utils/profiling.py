import functools
import logging
import os
from cProfile import Profile
from typing import Optional

logger = logging.getLogger(__name__)

PROFILING = int(os.getenv("PROFILING", 0))
CALLING_HOME = os.getcwd()
PROFILE_LOG_HOME = os.path.join(CALLING_HOME, "profiling_logs")

if PROFILING:
    from pyinstrument import Profiler


def profile(_func=None, *, name: Optional[str] = None):
    """
    Captures profiling information for the wrapped function using cProfile.

    Outputs a .prof file at PROFILE_LOG_HOME

    kwargs:
        name:   Name of the file output. Defaults to the name of the wrapped function.
    """

    def profile_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not PROFILING:
                logger.warning("Profiling turned off")
                return func(*args, **kwargs)

            tracer = Profile()
            tracer.enable()
            results = func(*args, **kwargs)
            tracer.disable()
            output_path = os.path.join(
                PROFILE_LOG_HOME, f"{name or func.__name__}.prof"
            )
            logger.info(f"Dumping stats: {output_path}")
            tracer.dump_stats(output_path)
            logger.info("Dumped stats")
            return results

        return wrapper

    if _func is None:
        return profile_decorator
    else:
        return profile_decorator(_func)


def pinstrument_profile(_func=None, *, name: Optional[str] = None):
    """
    Captures profiling information for the wrapped function using PyInstrument.

    Outputs a .text file at PROFILE_LOG_HOME

    kwargs:
        name:   Name of the file output. Defaults to the name of the wrapped function.
    """

    def profile_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not PROFILING:
                logger.warning("Profiling turned off")
                return func(*args, **kwargs)
            logger.info("Profiling turned on")
            tracer = Profiler(async_mode="disabled")
            tracer.start()
            results = func(*args, **kwargs)
            tracer.stop()
            output_path = os.path.join(PROFILE_LOG_HOME, f"{name or func.__name__}.txt")
            logger.info(f"Dumping stats: {output_path}")
            with open(output_path, "w") as writer:
                tracer.print(file=writer)
            logger.info("Dumped stats")
            return results

        return wrapper

    if _func is None:
        return profile_decorator
    else:
        return profile_decorator(_func)

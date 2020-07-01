import time
import logging
from typing import Callable
from typing import Optional


_PROGRESS_FN_TYPE = Callable[[int, Optional[int], bool], None]


def _default_progress_str(progress: int, total: Optional[int], final: bool):
    """
    Default progress string

    Args:
        progress: The progress so far as an integer.
        total: The total progress.
        final: Whether this is the final call.

    Returns:
        A formatted string representing the progress so far.
    """
    prefix = "completed " if final else ""
    if total is not None:
        percent = (100.0 * progress) / total
        return "%s%d / %d (%3d%%)" % (prefix, progress, total, int(percent))
    else:
        return "%s%d / %d" % (prefix, progress, progress)


class IntervalProgress:
    """
    A progress hook function that reports to output at a specified interval.
    """
    def __init__(self, interval: float = 1.0,
                 progress_str: _PROGRESS_FN_TYPE = _default_progress_str):
        self.interval = interval
        self.progress_str = progress_str
        self.last_update = time.time() - interval

    def __call__(self, progress: int, total: Optional[int], final: bool):
        if final or time.time() - self.last_update >= self.interval:
            self.progress(progress, total, final)
            self.last_update = time.time()

    def progress(self, progress: int, total: Optional[int], final: bool):
        """Processes the progress so far. Called only once per interval.

        Args:
            progress: The progress so far.
            total: The total to reach.
            final: Whether this is the final progress call.
        """
        raise NotImplementedError


class LoggingProgress(IntervalProgress):
    """
    An interval progress hook that reports to logging.info.
    """
    def progress(self, progress, total, final):
        logging.info(self.progress_str(progress, total, final))


class TerminalProgress(IntervalProgress):
    """
    An interval progress hook that writes to the terminal via print.
    """
    def progress(self, progress, total, final):
        print("\033[K" + self.progress_str(progress, total, final),
              end="\n" if final else "\r")

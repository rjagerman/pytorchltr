import time
import logging


def _default_progress_str(progress, total, final):
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
    def __init__(self, interval=1.0, progress_str=_default_progress_str):
        self.interval = interval
        self.progress_str = progress_str
        self.last_update = time.time() - interval

    def __call__(self, bytes_read, total_size, final):
        if final or time.time() - self.last_update >= self.interval:
            self.progress(bytes_read, total_size, final)
            self.last_update = time.time()

    def progress(self, progress, total, final):
        """Processes the progress so far. Called only once per interval.

        Args:
            progress (int): The progress so far.
            total (int, optional): The total to reach.
            final (bool): Whether this is the final progress call.
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

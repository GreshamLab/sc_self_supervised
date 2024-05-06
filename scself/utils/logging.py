import sys
import logging


class _Log:

    logger = None
    level = 40


if _Log.logger is None:

    _Log.logger = logging.Logger('scself')

    _logger_handler = logging.StreamHandler(sys.stderr)
    _logger_handler.setFormatter(
        logging.Formatter(
            '%(asctime)-15s %(levelno)s %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    )

    _Log.logger.addHandler(_logger_handler)


def verbose(flag):
    _Log.logger.setLevel(20 if flag else 40)


def log(*args, level=30, **kwargs):
    _Log.logger.log(level, *args, **kwargs)

import logging
import datetime
from tqdm import tqdm
import sys

def init_logger(module_name, outdir, tag="", level_console="info", level_file="debug"):
    level_dic = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET
    }
    if len(tag) == 0:
        tag = f"{datetime.datetime.now().strftime("%Y%m%d")}_{module_name}"
    
    fmt = logging.Formatter(
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt = "%Y%m%d-%H%M%S"
    )

    fh = logging.FileHandler(filename = outdir / f"{tag}_log.txt")
    fh.setLevel(level_dic[level_file])
    fh.setFormatter(fmt)

    th = TqdmLoggingHandler()
    th.setLevel(level_dic[level_console])
    th.setFormatter(fmt)

    logging.basicConfig(
        level=level_dic["notset"],
        handlers=[fh, th]
    )

    logger = logging.getLogger(module_name)
    return logger

class TqdmLoggingHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            tqdm.write(msg,file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)
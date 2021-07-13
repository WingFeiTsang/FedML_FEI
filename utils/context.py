import sys
import time
from contextlib import contextmanager
import threading

import traceback
from mpi4py import MPI

@contextmanager
def raise_MPI_error():
    import logging
    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        traceback.print_exception(*sys.exc_info())
        # logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
        logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
        MPI.COMM_WORLD.Abort()
    # else:
    #     traceback.print_exception(*sys.exc_info())
    #     logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
    #     MPI.COMM_WORLD.Abort()


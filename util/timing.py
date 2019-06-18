from contextlib import contextmanager
from time import process_time

__all__ = ['process_timer']


@contextmanager
def process_timer(task, msg='{task}: elapsed {elapsed} s', print_fn=print):
    start = process_time()
    yield
    elapsed = process_time() - start
    print_fn(msg.format(task=task, elapsed=elapsed))

import os

def get_max_workers(buffer=1):
    max_workers = max(1, os.cpu_count() - buffer)  # Leave buffer core free
    return max_workers

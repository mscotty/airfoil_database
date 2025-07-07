# airfoil_database/utilities/parallel_processing.py

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable, Any, Iterable

def parallel_map(func: Callable, items: Iterable, max_workers: int = None, 
                 show_progress: bool = True) -> List[Any]:
    """
    Execute a function over a list of items in parallel with progress reporting.
    
    Args:
        func: The function to apply to each item
        items: The list of items to process
        max_workers: Maximum number of worker processes (defaults to CPU count - 1)
        show_progress: Whether to show progress updates
        
    Returns:
        List of results from applying the function to each item
    """
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    items = list(items)  # Convert to list if it's not already
    total_items = len(items)
    results = []
    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            completed += 1
            result = future.result()
            
            if result is not None:
                results.append(result)
            
            # Show progress
            if show_progress and (completed % 10 == 0 or completed == total_items):
                elapsed = time.time() - start_time
                items_per_sec = completed / elapsed if elapsed > 0 else 0
                remaining = (total_items - completed) / items_per_sec if items_per_sec > 0 else 0
                
                print(f"Progress: {completed}/{total_items} ({completed/total_items*100:.1f}%) "
                      f"- {items_per_sec:.1f} items/sec - Est. remaining: {remaining:.1f}s")
    
    if show_progress:
        total_time = time.time() - start_time
        print(f"Completed {len(results)}/{total_items} items in {total_time:.2f} seconds "
              f"({len(results)/total_time:.1f} items/sec)")
    
    return results

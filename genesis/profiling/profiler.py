# profiler.py
"""
Unified profiler for timing code sections and functions.

Provides both manual section timing and function decorators with optional
detailed cProfile integration. Tracks individual iterations and computes
statistics (mean, std, min, max).
"""

import time
import cProfile
import pstats
import io
import math
from contextlib import contextmanager
from functools import wraps
from typing import List, Dict


class ProfileStats:
    """Statistics for a profiled section."""
    
    def __init__(self):
        self.times: List[float] = []
    
    def add_time(self, elapsed: float):
        """Add a timing measurement."""
        self.times.append(elapsed)
    
    @property
    def count(self) -> int:
        """Number of calls."""
        return len(self.times)
    
    @property
    def total(self) -> float:
        """Total time across all calls."""
        return sum(self.times)
    
    @property
    def mean(self) -> float:
        """Mean time per call."""
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    @property
    def std(self) -> float:
        """Standard deviation of times."""
        if len(self.times) < 2:
            return 0.0
        
        mean_val = self.mean
        variance = sum((t - mean_val) ** 2 for t in self.times) / (len(self.times) - 1)
        return math.sqrt(variance)
    
    @property
    def min(self) -> float:
        """Minimum time."""
        return min(self.times) if self.times else 0.0
    
    @property
    def max(self) -> float:
        """Maximum time."""
        return max(self.times) if self.times else 0.0


class Profiler:
    """Profiler for timing code sections with minimal overhead."""
    
    def __init__(self, enabled=True):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled (default: True)
        """
        self.enabled = enabled
        self.stats: Dict[str, ProfileStats] = {}
    
    @contextmanager
    def time(self, name):
        """
        Time a section of code.
        
        Args:
            name: Name for this timed section
            
        Example:
            with profiler.time("data_loading"):
                data = load_data()
        """
        if not self.enabled:
            yield
            return
        
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            
            # Store individual timing
            if name not in self.stats:
                self.stats[name] = ProfileStats()
            self.stats[name].add_time(elapsed)
    
    def print(self, show_stats=True):
        """
        Print timing summary sorted by total time (descending).
        
        Args:
            show_stats: Show detailed statistics (mean, std, min, max) (default: True)
        """
        if not self.enabled or not self.stats:
            return
        
        total_time = sum(s.total for s in self.stats.values())
        
        print(f"\nTotal: {total_time:.4f}s")
        
        if show_stats:
            # Print header
            print(f"{'Section':<30} {'Calls':>7} {'Total':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'%':>6}")
            print("-" * 105)
            
            # Print each section sorted by total time
            for name, stat in sorted(self.stats.items(), key=lambda x: x[1].total, reverse=True):
                pct = (stat.total / total_time * 100) if total_time > 0 else 0
                print(f"{name:<30} {stat.count:>7} "
                      f"{stat.total:>9.4f}s {stat.mean:>9.4f}s {stat.std:>9.4f}s "
                      f"{stat.min:>9.4f}s {stat.max:>9.4f}s {pct:>5.1f}%")
        else:
            # Simple output without statistics
            for name, stat in sorted(self.stats.items(), key=lambda x: x[1].total, reverse=True):
                pct = (stat.total / total_time * 100) if total_time > 0 else 0
                if stat.count > 1:
                    print(f"  {name:<30} {stat.total:>9.4f}s ({stat.count} calls)  ({pct:.1f}%)")
                else:
                    print(f"  {name:<30} {stat.total:>9.4f}s  ({pct:.1f}%)")
    
    def reset(self):
        """Clear all timing data."""
        self.stats.clear()
    
    @property
    def times(self) -> Dict[str, float]:
        """
        Backward compatibility: Return dict of section names to total times.
        
        Returns:
            Dictionary mapping section names to total times
        """
        return {name: stat.total for name, stat in self.stats.items()}


# Global profiler instance for use with decorator
_profiler = None


def get_profiler(enabled=True):
    """
    Get or create global profiler instance.
    
    Args:
        enabled: Whether profiling is enabled (default: True)
        
    Returns:
        Global Profiler instance
    """
    global _profiler
    if _profiler is None:
        _profiler = Profiler(enabled=enabled)
    return _profiler


def reset_profiler():
    """Reset global profiler (clear all timing data)."""
    global _profiler
    if _profiler is not None:
        _profiler.reset()


def enable_profiling():
    """Enable profiling globally."""
    global _profiler
    _profiler = Profiler(enabled=True)


def disable_profiling():
    """Disable profiling globally."""
    global _profiler
    _profiler = Profiler(enabled=False)


def profile_print(show_stats=True):
    """
    Print results from global profiler.
    
    Args:
        show_stats: Show detailed statistics (mean, std, min, max) (default: True)
    """
    p = get_profiler()
    p.print(show_stats=show_stats)


# ============================================================================
# UNIFIED DECORATOR
# ============================================================================

def profile(func=None, *, enabled=True, detailed=False, name=None, top_n=15, sort_by='cumulative'):
    """
    Unified decorator for profiling functions.
    
    Can be used for simple timing (fast, clean output) or detailed profiling
    (shows all function calls with cProfile).
    
    Args:
        enabled: Enable/disable profiling (default: True)
        detailed: Use cProfile for detailed analysis (default: False)
        name: Custom name for the profile (default: function name)
        top_n: Number of functions to show in detailed mode (default: 15)
        sort_by: Sort key for detailed mode: 'cumulative', 'time', 'calls' (default: 'cumulative')
    
    Usage:
        # Simple timing (recommended for daily use)
        @profile
        def my_function():
            ...
        
        # Detailed profiling (for deep analysis)
        @profile(detailed=True)
        def my_function():
            ...
        
        # Custom configuration
        @profile(name="Custom Name", top_n=20, sort_by='time')
        def my_function():
            ...
        
        # Disable specific function
        @profile(enabled=False)
        def my_function():
            ...
    
    Note:
        - Simple mode has minimal overhead (~5-10%)
        - Detailed mode has higher overhead but shows all internal calls
        - Use simple mode for daily profiling, detailed mode when investigating bottlenecks
        - Statistics (mean, std, min, max) are computed for functions called multiple times
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not enabled:
                return f(*args, **kwargs)
            
            section_name = name if name else f.__name__
            
            # Simple profiling (fast)
            if not detailed:
                p = get_profiler(enabled=True)
                with p.time(section_name):
                    return f(*args, **kwargs)
            
            # Detailed profiling with cProfile
            else:
                pr = cProfile.Profile()
                pr.enable()
                
                start = time.perf_counter()
                try:
                    retval = f(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    pr.disable()
                    
                    # Also record in simple profiler for consistency
                    p = get_profiler(enabled=True)
                    if section_name not in p.stats:
                        p.stats[section_name] = ProfileStats()
                    p.stats[section_name].add_time(elapsed)
                    
                    # Print detailed stats
                    s = io.StringIO()
                    ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
                    ps.print_stats(top_n)
                    
                    print(f"\n{'='*70}")
                    print(f"Detailed profile: {section_name}")
                    print(f"Total time: {elapsed:.4f}s")
                    print(f"{'='*70}")
                    print(s.getvalue())
                    print(f"{'='*70}\n")
                
                return retval
        
        return wrapper
    
    # Handle both @profile and @profile()
    if func is None:
        return decorator
    else:
        return decorator(func)
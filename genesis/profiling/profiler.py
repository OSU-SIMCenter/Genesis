# profiler.py
"""
Hierarchical profiler for timing code sections and functions.

Features:
- Hierarchical call stack tracking (manual instrumentation)
- Export to Speedscope JSON format (flame graphs, timeline)
- Console visualization with Aggregation (ASCII Tree, Rich Table)
- Flat "Hot Spot" analysis
- Filtering of negligible steps
- Browser visualization (Auto-open interactve Flame Graphs / Speedscope)
"""

import time
import json
import math
import sys
import tempfile
import webbrowser
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, FrozenSet
from functools import wraps
from pathlib import Path

# Sections excluded from active-compute % denominators (intentional pacing, not work).
EXCLUDED_FROM_ACTIVE_PCT: FrozenSet[str] = frozenset({
    "teleop_frame_pacing_sleep",
})

# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ProfileEvent:
    """Single profiling event with timing info."""
    name: str
    start: int  # nanoseconds from perf_counter_ns
    end: int = 0
    children: List['ProfileEvent'] = field(default_factory=list)

class ProfileStats:
    """Statistics for a profiled section (Backward Compatibility)."""
    
    def __init__(self):
        self._times: List[float] = []
    
    def add_time(self, elapsed: float):
        """Add a timing measurement."""
        self._times.append(elapsed)
    
    @property
    def count(self) -> int:
        return len(self._times)
    
    @property
    def total(self) -> float:
        return sum(self._times)
    
    @property
    def mean(self) -> float:
        return sum(self._times) / len(self._times) if self._times else 0.0
    
    @property
    def std(self) -> float:
        if len(self._times) < 2:
            return 0.0
        mean_val = self.mean
        variance = sum((t - mean_val) ** 2 for t in self._times) / (len(self._times) - 1)
        return math.sqrt(variance)
    
    @property
    def min(self) -> float:
        return min(self._times) if self._times else 0.0
    
    @property
    def max(self) -> float:
        return max(self._times) if self._times else 0.0


class Profiler:
    """Hierarchical profiler with advanced visualization and export."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.root = ProfileEvent("root", time.perf_counter_ns())
        self.stack = [self.root]
        self._cached_stats = None  # Cache for flat stats
        self._profiler_start_time = time.perf_counter_ns()  # Track when profiler was started/reset
    
    def reset(self):
        """Clear all timing data."""
        self._reset_internal_state()

    @contextmanager
    def time(self, name: str):
        """Profile a section of code."""
        if not self.enabled:
            yield
            return

        start_t = time.perf_counter_ns()
        event = ProfileEvent(name, start_t)
        
        # Add to current parent
        if self.stack:
            self.stack[-1].children.append(event)
        
        self.stack.append(event)
        try:
            yield
        finally:
            end_t = time.perf_counter_ns()
            event.end = end_t
            # Safety check: only pop if this is the expected event
            if self.stack and self.stack[-1] is event:
                self.stack.pop()
            
            # Invalidate stats cache since we added data
            self._cached_stats = None

    def stop(self):
        """Ensure root is closed (useful before printing/exporting)."""
        if self.root.end == 0:
            self.root.end = time.perf_counter_ns()

    @property
    def stats(self) -> Dict[str, ProfileStats]:
        """Aggregates the hierarchical tree into flat statistics (For Backward Compatibility)."""
        if self._cached_stats is not None:
            return self._cached_stats
            
        aggregated = {}

        def traverse(node: ProfileEvent):
            # Skip the artificial root
            if node.name != "root":
                if node.name not in aggregated:
                    aggregated[node.name] = ProfileStats()
                # Convert nanoseconds to seconds for backward compatibility
                aggregated[node.name].add_time((node.end - node.start) / 1e9)
            
            for child in node.children:
                traverse(child)

        traverse(self.root)
        self._cached_stats = aggregated
        return aggregated

    # ─────────────────────────────────────────────────────────────
    # TERMINAL VISUALIZATION
    # ─────────────────────────────────────────────────────────────

    @dataclass
    class _AggNode:
        name: str
        total: float = 0.0
        self_time: float = 0.0
        recursive_unprofiled: float = 0.0 # Cumulative unprofiled time (Self + Children's unprofiled)
        count: int = 0
        children: Dict[str, '_AggNode'] = field(default_factory=dict)

    def _aggregate_tree(self):
        """Aggregate raw timeline events into a cleaner call tree."""
        root_agg = self._AggNode("root")
        
        # Convert nanoseconds to seconds for display
        root_agg.total = (self.root.end - self.root.start) / 1e9

        def recurse(raw_node: ProfileEvent, agg_node: '_AggNode'):
            # Handle active nodes (end=0) by using root's end time (snapshot time)
            end_t = raw_node.end
            if end_t == 0:
                end_t = self.root.end
                
            dur_ns = end_t - raw_node.start
            
            # Calculate children duration similarly
            children_dur_ns = 0
            for c in raw_node.children:
                c_end = c.end
                if c_end == 0: c_end = self.root.end
                children_dur_ns += (c_end - c.start)

            self_dur_ns = dur_ns - children_dur_ns
            
            # Convert to seconds for _AggNode
            agg_node.total += dur_ns / 1e9
            agg_node.self_time += self_dur_ns / 1e9
            agg_node.count += 1
            
            for child in raw_node.children:
                if child.name not in agg_node.children:
                    agg_node.children[child.name] = self._AggNode(child.name)
                recurse(child, agg_node.children[child.name])

        for child in self.root.children:
            if child.name not in root_agg.children:
                root_agg.children[child.name] = self._AggNode(child.name)
            recurse(child, root_agg.children[child.name])
            
        # Post-process: Calculate recursive unprofiled time (Bottom-Up)
        def calc_recursive(node: '_AggNode'):
            if not node.children:
                # Leaf node: This IS a profiled section, so it contributes 0 to "Unprofiled"
                node.recursive_unprofiled = 0.0
                return

            child_unprof_sum = 0.0
            for child in node.children.values():
                calc_recursive(child)
                child_unprof_sum += child.recursive_unprofiled
            
            # Recursive unprofiled = My own glue code (self_time) + Children's unprofiled glue
            node.recursive_unprofiled = node.self_time + child_unprof_sum

        calc_recursive(root_agg)
            
        return root_agg

    def _collect_excluded_stats(self, root_agg: "_AggNode") -> Dict[str, float]:
        """Sum total time for sections excluded from active-compute reporting."""
        excluded: Dict[str, float] = {}

        def walk(node: "_AggNode") -> None:
            if node.name in EXCLUDED_FROM_ACTIVE_PCT:
                excluded[node.name] = excluded.get(node.name, 0.0) + node.total
                return
            for child in node.children.values():
                walk(child)

        for child in root_agg.children.values():
            walk(child)
        return excluded

    def _sum_non_excluded_self_time(self, root_agg: "_AggNode") -> float:
        """Sum self-time for all profiled sections except excluded pacing sleep."""
        total = 0.0

        def walk(node: "_AggNode") -> None:
            nonlocal total
            if node.name in EXCLUDED_FROM_ACTIVE_PCT:
                return
            if node.name != "root":
                total += node.self_time
            for child in node.children.values():
                walk(child)

        for child in root_agg.children.values():
            walk(child)
        return total

    def _wall_and_active_totals(self, root_agg: "_AggNode") -> tuple[float, float, Dict[str, float]]:
        elapsed_ns = time.perf_counter_ns() - self._profiler_start_time
        wall_time = elapsed_ns / 1_000_000_000
        excluded_stats = self._collect_excluded_stats(root_agg)

        # Use non-excluded self-time as the active-compute denominator.
        # Do NOT subtract excluded node.total from sum(root children.total): excluded
        # sections are usually nested (e.g. pacing under teleop_step), while root
        # children only count top-level markers. With interleaved asyncio tasks that
        # mismatch can make active_time == 0 and collapse all % to zero.
        active_time = self._sum_non_excluded_self_time(root_agg)
        return wall_time, active_time, excluded_stats

    def _print_excluded_footer(
        self,
        console,
        use_rich: bool,
        wall_time: float,
        active_time: float,
        excluded_stats: Dict[str, float],
        idle_time: float,
    ) -> None:
        for name in sorted(excluded_stats):
            excluded_ms = excluded_stats[name] * 1000
            wall_pct = (excluded_stats[name] / wall_time * 100) if wall_time > 0 else 0.0
            label = f" [Excluded: {name}]"
            line = f"{label:<30} (excluded)  {excluded_ms:>9.1f}ms  ({wall_pct:.1f}% wall)"
            if use_rich:
                console.print(f"[dim italic]{line}[/]", highlight=False)
            else:
                print(line)

        if active_time > 0.001:
            active_ms = active_time * 1000
            wall_pct = (active_time / wall_time * 100) if wall_time > 0 else 0.0
            line = f" [Active Compute Time]        (denominator) {active_ms:>9.1f}ms  ({wall_pct:.1f}% wall)"
            if use_rich:
                console.print(f"[dim]{line}[/]", highlight=False)
            else:
                print(line)

        if idle_time > 0.001:
            idle_ms = idle_time * 1000
            if use_rich:
                console.print(
                    f"[dim italic] [Idle/Inactive Time]         (excluded)  {idle_ms:>9.1f}ms[/]",
                    highlight=False,
                )
            else:
                print(f" [Idle/Inactive Time]         (excluded)  {idle_ms:>9.1f}ms")

    def _process_children_for_display(
        self,
        node: "_AggNode",
        min_pct: float,
        grand_total: float,
        exclude_names: Optional[Set[str]] = None,
    ):
        """
        Sorts children and groups small ones into 'Others'.
        Returns list of nodes to display.
        """
        children = node.children.values()
        if exclude_names:
            children = [c for c in children if c.name not in exclude_names]
        sorted_children = sorted(children, key=lambda c: c.total, reverse=True)
        
        if min_pct <= 0 or grand_total <= 0:
            return sorted_children

        kept = []
        others = self._AggNode(name="(Others)")
        
        for c in sorted_children:
            pct = (c.total / grand_total) * 100
            if pct >= min_pct:
                kept.append(c)
            else:
                others.total += c.total
                others.self_time += c.self_time 
                others.recursive_unprofiled += c.recursive_unprofiled
                others.count += c.count

        if others.count > 0:
            others.name = f"(Others: {others.count})"
            kept.append(others)
            
        return kept

    def print(self, show_stats=True, min_pct=0.0):
        """
        Backward compatible print. Prefers rich_table if available.
        min_pct: Minimum absolute percentage to display a section (filters noise).
        """
        if not self.enabled: return
        self.stop()
        
        try:
            import rich
            self.rich_table(min_pct=min_pct)
            return
        except ImportError:
            pass
            
        # Fallback
        self.print_flat(sort_by="total", min_pct=min_pct)

    def _get_console(self):
        try:
            from rich.console import Console
            return Console(), True
        except ImportError:
            return None, False

    def print_flat(self, sort_by="self", min_pct=0.0):
        """
        Print a flat list of 'Elementary Steps' (aggregated self-time).
        sort_by: 'self' (default, shows hot spots) or 'total'.
        """
        if not self.enabled: return
        self.stop()
        
        console, use_rich = self._get_console()
        
        # Flatten by name
        flat_stats: Dict[str, Profiler._AggNode] = {}
        
        root_agg = self._aggregate_tree()

        wall_time, active_time, excluded_stats = self._wall_and_active_totals(root_agg)
        profiled_time = sum(c.total for c in root_agg.children.values())
        idle_time = wall_time - profiled_time
        total_time = active_time
        
        def collect(node: Profiler._AggNode):
            for child in node.children.values():
                collect(child)

            # Aggregate self-time from every profiled node (not just leaves).
            # Parent blocks like teleop_frame_pacing_sleep have no children but
            # previously inflated [Unprofiled] when grouped under (Others).
            if node.name == "root" or node.self_time <= 0:
                return

            if node.name not in flat_stats:
                flat_stats[node.name] = self._AggNode(node.name)

            target = flat_stats[node.name]
            target.total += node.self_time
            target.self_time += node.self_time
            target.count += node.count
        
        # Collect from children of root
        for child in root_agg.children.values():
            collect(child)
            
        items = [
            node for node in flat_stats.values()
            if node.name not in EXCLUDED_FROM_ACTIVE_PCT
        ]
        items.sort(key=lambda x: x.total, reverse=True)
        
        # Remaining glue: profiled tree time not attributed to any section self-time
        sum_self = sum(x.self_time for x in items)
        unprofiled_time = max(0.0, total_time - sum_self)
        
        if unprofiled_time > 0 and total_time > 0:
            unprof_node = self._AggNode(" [Unprofiled (Glue Code)] ")
            unprof_node.total = unprofiled_time
            unprof_node.self_time = unprofiled_time
            unprof_node.count = 1 # Abstract count
            
            # Insert at appropriate position
            inserted = False
            for i, item in enumerate(items):
                if unprof_node.total > item.total:
                    items.insert(i, unprof_node)
                    inserted = True
                    break
            if not inserted:
                items.append(unprof_node)

        value_getter = lambda x: x.total
        title = "Profile Results (Hot Spots - Self Time, Active Compute)"
            
        print(f"\n--- {title} ---")
        if use_rich:
            print(f"{'Section':<30} {'Calls':>7} {'Total':>10} {'%':>8}   {'Graph'}")
            print("-" * 105)
        else:
            print(f"{'Section':<30} {'Calls':>7} {'Total':>10} {'%':>8}   {'Graph'}")
            print("-" * 105)
        
        # Softened Heatmap (Red -> Orange -> Yellow -> Green)
        colors_heat = ["red", "orange1", "dark_orange", "gold1", "yellow", "chartreuse1", "green3", "green"]
        
        # Determine max value for relative scaling
        max_val = 0.0
        if items:
            max_val = value_getter(items[0])

        # Separate items above and below cutoff, plus build "Others" aggregate
        above_cutoff = []
        others = self._AggNode("(Others)")
        others_count = 0
        
        for node in items:
            val = value_getter(node)
            pct = (val / total_time * 100) if total_time > 0 else 0
            
            if pct >= min_pct:
                above_cutoff.append(node)
            else:
                others.total += node.total
                others.self_time += node.self_time
                others.count += node.count
                others_count += 1
        
        # Finalize "Others" name with count
        if others_count > 0:
            others_pct = (others.total / total_time * 100) if total_time > 0 else 0
            others.name = f"(Others: {others_count} items, {others_pct:.1f}%)"
        
        for node in above_cutoff:
            val = value_getter(node)
            pct = (val / total_time * 100) if total_time > 0 else 0
            
            # Simple ASCII bar
            bar_len = 25
            filled = int(pct / 100 * bar_len)
            bar_str = "█" * filled + "░" * (bar_len - filled)
            
            if use_rich:
                # Color logic
                if max_val > 0:
                    relative = val / max_val
                else:
                    relative = 0
                
                idx = int((1.0 - relative) * (len(colors_heat) - 1))
                idx = max(0, min(len(colors_heat) - 1, idx))
                color = colors_heat[idx]
                
                console.print(f"[{color}]{node.name:<30} {node.count:>7} "
                      f"{node.total*1000:>9.1f}ms {pct:>7.1f}%   {bar_str}[/]", 
                      highlight=False)
            else:
                 print(f"{node.name:<30} {node.count:>7} "
                      f"{node.total*1000:>9.1f}ms {pct:>7.1f}%   {bar_str}")
        
        # Print "Others" at the bottom if there are any
        if others_count > 0:
            others_pct = (others.total / total_time * 100) if total_time > 0 else 0
            bar_len = 25
            filled = int(others_pct / 100 * bar_len)
            bar_str = "█" * filled + "░" * (bar_len - filled)
            
            if use_rich:
                console.print(f"[dim]{others.name:<30} {others.count:>7} "
                      f"{others.total*1000:>9.1f}ms {others_pct:>7.1f}%   {bar_str}[/]", 
                      highlight=False)
            else:
                print(f"{others.name:<30} {others.count:>7} "
                      f"{others.total*1000:>9.1f}ms {others_pct:>7.1f}%   {bar_str}")
        
        print("---------------------------------------------------------------------------------------------------------")

        self._print_excluded_footer(console, use_rich, wall_time, active_time, excluded_stats, idle_time)

        print("")

    def print_tree(self, min_pct=0.0):
        """Simple terminal tree with ASCII bars (Aggregated)."""
        if not self.enabled: return
        self.stop()
        
        console, use_rich = self._get_console()

        root_agg = self._aggregate_tree()
        wall_time, active_time, excluded_stats = self._wall_and_active_totals(root_agg)
        profiled_time = sum(c.total for c in root_agg.children.values())
        idle_time = wall_time - profiled_time
        total_time = active_time

        print("\n--- Profile Tree (Aggregated, Active Compute) ---")
        
        # Consistent color cycle by depth
        colors = ["bold cyan", "green", "yellow", "magenta", "blue"]
        
        def walk(node: '_AggNode', prefix: str, is_last: bool, depth: int):
            pct = (node.total / total_time * 100) if total_time > 0 else 0
            
            # 1. Absolute Cumulative Unprofiled (Recursive / Global Total)
            # This accounts for the node's glue code PLUS all its children's glue code
            abs_cum_unprof_pct = (node.recursive_unprofiled / total_time) * 100 if total_time > 0 else 0.0

            # 2. Relative Local Unprofiled (Self / Node Total)
            # This accounts for how much of THIS node is just glue code vs delegated to children
            if node.total > 0:
                rel_unprof_pct = (node.self_time / node.total) * 100
            else:
                rel_unprof_pct = 0.0
            
            # Connector
            connector = "└── " if is_last else "├── "
            
            # Info string
            # Only show unprofiled info if it's non-negligible
            unprof_info = ""
            if abs_cum_unprof_pct > 0.1:
                # For non-leaves, show both Absolute Cumulative and Relative
                if node.children:
                    unprof_info = f" [Unprofiled: {abs_cum_unprof_pct:.1f}% Abs Cum, {rel_unprof_pct:.1f}% Rel]"
                else:
                    # For leaves, Relative is technically 100% (all self time), which is confusing.
                    # Just show Absolute Cumulative (which is 0.0 for leaves properly).
                    # Actually, if it's 0.0, we just skip it.
                    if abs_cum_unprof_pct > 0.1:
                         unprof_info = f" [Unprofiled: {abs_cum_unprof_pct:.1f}% Abs Cum]"

            count_info = f" ({node.count}x)" if node.count > 1 else ""
            
            # Bar 
            bar_len = 30
            filled = int(pct / 100 * bar_len)
            bar_str = "█" * filled + "░" * (bar_len - filled)
            
            # Color
            if use_rich:
                color = colors[depth % len(colors)]
                # Reset prefix and connector to dim, color the name and info
                console.print(f"[dim]{prefix}{connector}[/][{color}]{node.name}{count_info}: {node.total*1000:.1f}ms ({pct:.1f}%){unprof_info}[/]", highlight=False)
                
                # Bar Line
                bar_prefix = prefix + ("    " if is_last else "│   ")
                console.print(f"[dim]{bar_prefix}    [/][{color}]{bar_str}[/]", highlight=False)
            else:
                print(f"{prefix}{connector}{node.name}{count_info}: {node.total*1000:.1f}ms ({pct:.1f}%){unprof_info}")
                bar_prefix = prefix + ("    " if is_last else "│   ")
                print(f"{bar_prefix}    {bar_str}")
            
            # Prepare children
            children = self._process_children_for_display(
                node, min_pct, total_time, exclude_names=EXCLUDED_FROM_ACTIVE_PCT
            )

            # New prefix for children
            if use_rich:
                # We need clean string for prefix logic, stripped of colors
                child_prefix = prefix + ("    " if is_last else "│   ")
            else:
                child_prefix = bar_prefix # same as above

            for i, child in enumerate(children):
                walk(child, child_prefix, i == len(children) - 1, depth + 1)

        # Children of root
        children = self._process_children_for_display(
            root_agg, min_pct, total_time, exclude_names=EXCLUDED_FROM_ACTIVE_PCT
        )
        for i, child in enumerate(children):
            walk(child, "", i == len(children) - 1, 0)

        print("---------------------------------")
        self._print_excluded_footer(console, use_rich, wall_time, active_time, excluded_stats, idle_time)
        print("")

    def rich_table(self, min_pct=0.0):
        """Rich library table view (Aggregated)."""
        if not self.enabled: return
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            # Fallback
            self.print_tree(min_pct)
            return

        self.stop()
        console = Console()
        root_agg = self._aggregate_tree()
        wall_time, active_time, excluded_stats = self._wall_and_active_totals(root_agg)
        profiled_time = sum(c.total for c in root_agg.children.values())
        idle_time = wall_time - profiled_time
        total_time = active_time

        table = Table(title=f"Profile Results (Active Compute, >{min_pct}%)")
        table.add_column("Section", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Unprof (Abs Cum)", justify="right") 
        table.add_column("Unprof (Rel)", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Visualization", width=30)
        
        # Colors depth cycle
        colors = ["bold cyan", "green", "yellow", "magenta", "blue"]

        def build_table(node: '_AggNode', depth):
            pct = (node.total / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(pct / 3)
            
            # Indentation with guides
            indent = "[dim]│   [/]" * depth
            
            # Color name by depth
            style_name = colors[depth % len(colors)]
            # We strip 'bold' for the bar color usually, but keeping it simple: just use the color name
            # e.g. "bold cyan" -> "cyan"
            color_only = style_name.replace("bold ", "")
            
            # 1. Absolute Cumulative Unprofiled
            abs_cum_unprof_pct = (node.recursive_unprofiled / total_time * 100) if total_time > 0 else 0.0
            
            # 2. Relative Local Unprofiled (Hide for leaves since they're 100% self)
            rel_unprof_str = "-"
            if node.children and node.total > 0:
                rel_unprof_pct = (node.self_time / node.total) * 100
                rel_unprof_str = f"{rel_unprof_pct:.1f}%"

            # Apply color to all columns
            table.add_row(
                f"{indent}[{style_name}]{node.name}[/]",
                f"[{style_name}]{node.count}[/]",
                f"[{style_name}]{node.total*1000:.1f}ms[/]",
                f"[{style_name}]{node.recursive_unprofiled*1000:.1f}ms ({abs_cum_unprof_pct:.1f}%)[/]",
                f"[{style_name}]{rel_unprof_str}[/]", 
                f"[{style_name}]{pct:.1f}%[/]",
                f"[{color_only}]{bar}[/]" 
            )
            
            children_to_show = self._process_children_for_display(
                node, min_pct, total_time, exclude_names=EXCLUDED_FROM_ACTIVE_PCT
            )
            for child in children_to_show:
                build_table(child, depth + 1)

        children = self._process_children_for_display(
            root_agg, min_pct, total_time, exclude_names=EXCLUDED_FROM_ACTIVE_PCT
        )
        for child in children:
            build_table(child, 0)

        console.print(table)
        self._print_excluded_footer(console, True, wall_time, active_time, excluded_stats, idle_time)
        print("")

    # ─────────────────────────────────────────────────────────────
    # BROWSER VISUALIZATION (Auto-open)
    # ─────────────────────────────────────────────────────────────

    def _open_html(self, html, name_suffix):
        path = Path(tempfile.gettempdir()) / f"profile_{name_suffix}.html"
        path.write_text(html, encoding='utf-8')
        try:
            webbrowser.open(f"file://{path}")
            print(f"[Profiler] Opened visualization: file://{path}")
        except Exception:
            print(f"[Profiler] Saved visualization to: {path}")

    def _b64(self, s):
        import base64
        return base64.b64encode(s.encode()).decode()

    def open_flame(self):
        """Generate HTML flame graph and auto-open in browser."""
        if not self.enabled: return
        self.stop()
        
        def to_dict(node):
            dur_ns = node.end - node.start
            children_dur_ns = sum(c.end - c.start for c in node.children)
            # Convert nanoseconds to microseconds for flame graph value
            self_val = int((dur_ns - children_dur_ns) / 1000)
            return {
                "name": node.name,
                "value": self_val if self_val > 0 else 0, 
                "children": [to_dict(c) for c in node.children]
            }
        
        data = {
            "name": "root",
            "value": 0,
            "children": [to_dict(c) for c in self.root.children]
        }
        
        html = f'''<!DOCTYPE html>
<html><head>
<style>body {{ margin: 0; font-family: system-ui; }} #chart {{ height: 100vh; }}</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3-flame-graph@4.1.3/dist/d3-flamegraph.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/d3-flame-graph@4.1.3/dist/d3-flamegraph.css">
</head><body>
<div id="chart"></div>
<script>
var data = {json.dumps(data)};
var chart = flamegraph().width(window.innerWidth).cellHeight(20)
    .tooltip(d3.select("body").append("div").style("position","absolute")
        .style("background","#333").style("color","#fff").style("padding","5px")
        .style("border-radius","3px").style("font-size","12px").style("display","none"))
    .setLabelHandler(function(d) {{
        var ms = (d.value / 1000).toFixed(2);
        return d.data.name + " (" + ms + "ms)";
    }});
d3.select("#chart").datum(data).call(chart);
</script></body></html>'''
        self._open_html(html, "flamegraph")

    def open_speedscope(self):
        """Generate speedscope-compatible view and auto-open."""
        if not self.enabled: return
        self.stop()
        
        profile_data = self._build_speedscope_data()
        
        html = f'''<!DOCTYPE html>
<html><head><title>Profile</title></head>
<body style="margin:0">
<script>window.speedscopeConfig = {{ profileURL: "data:application/json;base64,{self._b64(json.dumps(profile_data))}" }}</script>
<script src="https://cdn.jsdelivr.net/npm/speedscope@1.20.0/dist/release/index.js"></script>
</body></html>'''
        self._open_html(html, "speedscope")

    # ─────────────────────────────────────────────────────────────
    # EXPORT (Manual)
    # ─────────────────────────────────────────────────────────────

    def _build_speedscope_data(self) -> dict:
        """Build speedscope-compatible data structure from profile tree."""
        frames, frame_idx, samples, weights = [], {}, [], []
        
        def get_idx(name):
            if name not in frame_idx:
                frame_idx[name] = len(frames)
                frames.append({"name": name})
            return frame_idx[name]
        
        def walk(node, stack):
            idx = get_idx(node.name)
            new_stack = stack + [idx]
            children_time_ns = sum(c.end - c.start for c in node.children)
            self_time_ns = (node.end - node.start) - children_time_ns
            # Threshold: 100 nanoseconds (was 1e-7 seconds = 100ns)
            if self_time_ns > 100:
                samples.append(new_stack[:])
                weights.append(self_time_ns)  # Already in nanoseconds
            for c in node.children:
                walk(c, new_stack)
        
        for c in self.root.children:
            walk(c, [])
        
        return {
            "$schema": "https://www.speedscope.app/file-format-schema.json",
            "shared": {"frames": frames},
            "profiles": [{
                "type": "sampled", "name": "profile", "unit": "nanoseconds",
                "startValue": 0, "endValue": sum(weights),
                "samples": samples, "weights": weights
            }]
        }

    def to_speedscope(self) -> dict:
        """Export to speedscope JSON format (raw dict)."""
        self.stop()
        return self._build_speedscope_data()

    def save_speedscope(self, path: str):
        """Save speedscope JSON to file."""
        data = self.to_speedscope()
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Profiler] Saved Speedscope profile to: {path}")

# ─────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────

_profiler = None

def get_profiler(enabled=True):
    global _profiler
    if _profiler is None:
        _profiler = Profiler(enabled=enabled)
    elif _profiler.enabled != enabled:
        # Reset stale data when changing enabled state to avoid corruption
        _profiler.reset()
        _profiler.enabled = enabled
    return _profiler

def reset_profiler():
    if _profiler is not None:
        _profiler.reset()

def enable_profiling():
    get_profiler(enabled=True)

def disable_profiling():
    get_profiler(enabled=False)

def profile_print(show_stats=True):
    if _profiler:
        _profiler.print(show_stats)

def profile(func=None, *, enabled=True, name=None):
    """Decorator for profiling functions using the global profiler."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            p = get_profiler(enabled=enabled)
            sec_name = name or f.__name__
            with p.time(sec_name):
                return f(*args, **kwargs)
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
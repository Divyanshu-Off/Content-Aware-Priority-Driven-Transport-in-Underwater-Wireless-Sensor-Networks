"""Performance Profiling and Monitoring."""
from typing import Dict, List, Optional
import time
from dataclasses import dataclass, field

@dataclass
class PerformanceMetrics:
    """Stores performance metrics."""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0

class PerformanceProfiler:
    """Profiles and monitors performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.active_timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.active_timers[operation] = time.time()
    
    def stop_timer(self, operation: str) -> Optional[float]:
        """Stop timing and record duration."""
        if operation not in self.active_timers:
            return None
        
        duration = time.time() - self.active_timers[operation]
        del self.active_timers[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = PerformanceMetrics(function_name=operation)
        
        metric = self.metrics[operation]
        metric.call_count += 1
        metric.total_time += duration
        metric.min_time = min(metric.min_time, duration)
        metric.max_time = max(metric.max_time, duration)
        metric.avg_time = metric.total_time / metric.call_count
        
        return duration
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict:
        """Get performance metrics."""
        if operation:
            return self.metrics.get(operation, None)
        return self.metrics
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        return {
            'total_operations': len(self.metrics),
            'total_calls': sum(m.call_count for m in self.metrics.values()),
            'total_time': sum(m.total_time for m in self.metrics.values()),
            'operations': {k: v.__dict__ for k, v in self.metrics.items()}
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.active_timers.clear()

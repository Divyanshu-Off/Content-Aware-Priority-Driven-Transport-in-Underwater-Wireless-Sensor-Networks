"""Event-driven simulation scheduler for UWSN."""

import heapq
from typing import Callable, Any, List
from dataclasses import dataclass, field

@dataclass(order=True)
class Event:
    """Represents a simulation event."""
    time: float
    priority: int = field(compare=False)
    event_type: str = field(compare=False)
    callback: Callable = field(compare=False)
    data: Any = field(compare=False, default=None)
    
class EventScheduler:
    """Manages discrete event simulation."""
    
    def __init__(self):
        self.event_queue = []
        self.current_time = 0.0
        self.event_count = 0
        
    def schedule_event(self, delay: float, event_type: str, 
                      callback: Callable, data: Any = None, priority: int = 0):
        """Schedule an event to occur after delay."""
        event_time = self.current_time + delay
        event = Event(event_time, priority, event_type, callback, data)
        heapq.heappush(self.event_queue, event)
        
    def schedule_at(self, time: float, event_type: str,
                   callback: Callable, data: Any = None, priority: int = 0):
        """Schedule an event to occur at specific time."""
        event = Event(time, priority, event_type, callback, data)
        heapq.heappush(self.event_queue, event)
        
    def get_next_event(self) -> Event:
        """Get the next event from queue."""
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None
        
    def run(self, until: float = None):
        """Run simulation until specified time or queue is empty."""
        while self.event_queue:
            event = self.get_next_event()
            
            if until is not None and event.time > until:
                # Put event back and stop
                heapq.heappush(self.event_queue, event)
                self.current_time = until
                break
                
            self.current_time = event.time
            self.event_count += 1
            
            # Execute event callback
            try:
                event.callback(event.data)
            except Exception as e:
                print(f"Error executing event {event.event_type}: {e}")
                
    def clear(self):
        """Clear all scheduled events."""
        self.event_queue = []
        self.current_time = 0.0
        self.event_count = 0
        
    def get_pending_events(self) -> int:
        """Get number of pending events."""
        return len(self.event_queue)
    
    def peek_next_time(self) -> float:
        """Peek at the time of next event without removing it."""
        if self.event_queue:
            return self.event_queue[0].time
        return None

"""Priority boost checking and anomaly detection."""

class PriorityBoostChecker:
    """Check and manage priority boost conditions."""
    
    ANOMALY_THRESHOLD = 3
    BOOST_COOLDOWN = 10
    
    def __init__(self):
        self.boost_count = 0
        self.last_boost_time = 0
        self.anomalies = []
    
    def check_for_anomaly(self, pkt, is_anomalous: bool) -> bool:
        """Check if packet exhibits anomalous behavior."""
        if is_anomalous:
            self.anomalies.append(id(pkt))
            if len(self.anomalies) >= self.ANOMALY_THRESHOLD:
                return True
        return False
    
    def apply_priority_boost(self, pkt) -> bool:
        """Apply priority boost if conditions met."""
        if len(self.anomalies) >= self.ANOMALY_THRESHOLD:
            self.boost_count += 1
            pkt.priority_boost = True
            self.anomalies.clear()
            return True
        return False

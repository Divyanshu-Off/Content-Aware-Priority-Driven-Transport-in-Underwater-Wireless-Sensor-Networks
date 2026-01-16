"""Error Handling and Recovery Mechanisms."""
from typing import Optional, Callable, Any
from enum import Enum
import traceback
import logging

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3

class ErrorHandler:
    """Handles errors and implements recovery mechanisms."""
    
    def __init__(self, log_file: str = 'uwsn_protocol.log'):
        self.error_count = 0
        self.error_history = []
        self.recovery_strategies = {}
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def handle_error(self, error: Exception, 
                     severity: ErrorSeverity = ErrorSeverity.ERROR,
                     context: Optional[dict] = None) -> bool:
        """Handle an error with appropriate logging and recovery."""
        self.error_count += 1
        
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'severity': severity.name,
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        if severity == ErrorSeverity.CRITICAL:
            logging.critical(f"Critical error: {error_info}")
            return False
        elif severity == ErrorSeverity.ERROR:
            logging.error(f"Error: {error_info}")
        elif severity == ErrorSeverity.WARNING:
            logging.warning(f"Warning: {error_info}")
        else:
            logging.info(f"Info: {error_info}")
        
        return True
    
    def register_recovery(self, error_type: type, 
                         recovery_fn: Callable) -> None:
        """Register a recovery function for specific error type."""
        self.recovery_strategies[error_type] = recovery_fn
    
    def attempt_recovery(self, error: Exception) -> Optional[Any]:
        """Attempt to recover from error using registered strategy."""
        error_type = type(error)
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error)
            except Exception as recovery_error:
                logging.error(f"Recovery failed: {recovery_error}")
        return None

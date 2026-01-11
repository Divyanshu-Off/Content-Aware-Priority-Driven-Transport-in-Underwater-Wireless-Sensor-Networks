"""UWSN Protocol Module - Packet and Node Classes

This module implements the core protocol logic for the UWSN Priority-Driven
Transport system, including:
  - Packet class with content-aware priority mapping
  - Node class with priority-based queuing and energy management
  - Energy consumption models

Reference: docs/PHASE_1_DESIGN.md and docs/PHASE_2_IMPLEMENTATION.md
"""

from .packet import Packet
from .node import Node
from . import node_energy

__all__ = [
    'Packet',
    'Node',
    'node_energy',
]

__version__ = '0.1.0'
__author__ = 'UWSN Project'

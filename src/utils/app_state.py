# src/utils/app_state.py
import threading
from collections import deque
from typing import Optional
import numpy as np
from collections import deque

class AppState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.is_running     = False
        self.static_label   = "—"
        self.static_conf    = 0.0
        self.motion_label   = "—"
        self.motion_conf    = 0.0
        self.active_profile = "—"
        self.fps            = 0.0
        self.stop_event     = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.command_log: deque = deque(maxlen=6)  # add this
state = AppState()
#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import threading
import time
from enum import Enum
from typing import Optional, Callable, Any

class VectorDBType(str, Enum):
    FAISS = "FAISS"
    MILVUS = "MILVUS"

class SearchType(str, Enum):
    COSINE = "COSINE"

class TimeUtil:
    """single-threaded timer utility class"""

    def __init__(self, interval: float, callback: Callable[[], Any], **kwargs):
        self.interval = interval
        self.callback = callback
        self.callback_kwargs = kwargs
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def _run_loop(self) -> None:
        """single-thread run loop"""
        next_run_time = time.time() + self.interval

        with self._running:
            current_time = time.time()

            if current_time >= next_run_time:
                # execute callback
                try:
                    print(f"start executing scheduled tasks")
                    if self.callback_kwargs:
                        self.callback(**self.callback_kwargs)
                    else:
                        self.callback()
                except Exception as e:
                    print(f"callback execution exception: {e}")

                next_run_time += self.interval

            sleep_time = max(0.1, next_run_time - time.time())
            time.sleep(min(sleep_time, 0.1))

    def __enter__(self):
        """support context manager"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """automatically stop when exiting context"""
        self.stop()

    def is_running(self) -> bool:
        """check if the timer is running"""
        with self._lock:
            return self._running

    def __del__(self):
        if hasattr(self, '_running' and self._running):
            self.stop()
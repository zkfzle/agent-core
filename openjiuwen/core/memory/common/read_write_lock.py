#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import threading


class ReadWriteLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False

    class _ReadLock:
        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            self.outer.acquire_read()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.outer.release_read()

    class _WriteLock:
        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            self.outer.acquire_write()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.outer.release_write()

    def read_lock(self):
        return ReadWriteLock._ReadLock(self)

    def write_lock(self):
        return ReadWriteLock._WriteLock(self)

    def acquire_read(self):
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def release_write(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()

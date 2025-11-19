#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import threading


class DataIdManager:
    SEPARATOR = "\x1F"
    ID_KEY = f'id_manager{SEPARATOR}next_id'

    def __init__(self, kv_store):
        self.lock = threading.Lock()
        self.kv_store = kv_store
        self.next_id = self.kv_store.get(DataIdManager.ID_KEY)
        if not self.next_id:
            self.next_id = 0
        self.next_id = int(self.next_id)

    def generate_next_id(self) -> int:
        """Generate a unique ID and store it in the KV"""
        with self.lock:
            ret = self.next_id
            self.next_id += 1
            self.kv_store.set(DataIdManager.ID_KEY, str(self.next_id))
            return ret

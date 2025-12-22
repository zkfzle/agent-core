#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import unittest

from openjiuwen.core.memory.common.crypto import encrypt, decrypt


class TestCrypto(unittest.TestCase):
    def test_encrypt(self):
        test_key = "1234567890abcdef"
        test_data = "hello, 我叫张三, xixi"
        encrypto_data, iv = encrypt(test_key, test_data)
        decrypto_data = decrypt(test_key, encrypto_data, iv)
        self.assertEqual(decrypto_data, test_data)

    def test_key_error(self):
        test_key = "1234567890abc"
        test_data = "你好, 我叫李四"
        with self.assertRaises(ValueError):
            encrypt(test_key, test_data)

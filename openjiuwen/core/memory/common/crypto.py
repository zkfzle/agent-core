#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import secrets
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

IV_LENGTH = 16
BIT_LENGTH = 8
PAD_STYLE = 'iso7816'
AES_KEY_LENGTH = 16


def encrypt(key: str, plaintext: str):
    aes_key = key.encode(encoding='utf-8')
    if len(aes_key) != AES_KEY_LENGTH:
        raise ValueError(f'Wrong key length: {len(aes_key)}, expected {AES_KEY_LENGTH}')
    plaintext_bytes = plaintext.encode(encoding="utf-8")
    random_instance = secrets.SystemRandom()
    iv = bytes([random_instance.getrandbits(BIT_LENGTH) for _ in range(0, IV_LENGTH)])
    padded_plaintext = pad(data_to_pad=plaintext_bytes, block_size=AES.block_size, style=PAD_STYLE)
    cipher = AES.new(key=aes_key, mode=AES.MODE_CBC, iv=iv)
    cipher_text = cipher.encrypt(padded_plaintext)
    return [cipher_text.hex(), iv.hex()]


def decrypt(key: str, ciphertext: str, iv: str):
    ciphertext_bytes = bytes.fromhex(ciphertext)
    iv_bytes = bytes.fromhex(iv)
    aes_key = key.encode(encoding='utf-8')
    if len(aes_key) != AES_KEY_LENGTH:
        raise ValueError(f'Wrong key length: {len(aes_key)}, expected {AES_KEY_LENGTH}')

    if len(ciphertext_bytes) % AES.block_size != 0:
        raise ValueError(f"Wrong cipher text length: {len(ciphertext_bytes)}")

    if len(iv_bytes) != IV_LENGTH:
        raise ValueError(f"Wrong iv length: {len(iv_bytes)}")

    cipher = AES.new(key=aes_key, mode=AES.MODE_CBC, iv=iv_bytes)
    plaintext_bytes = cipher.decrypt(ciphertext_bytes)
    unpadded_plaintext = unpad(plaintext_bytes, AES.block_size, style=PAD_STYLE)
    return unpadded_plaintext.decode(encoding="utf-8")

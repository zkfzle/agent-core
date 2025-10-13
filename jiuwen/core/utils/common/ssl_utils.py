#!/usr/bin/python3.11
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved
import os
import ssl

from requests.adapters import HTTPAdapter


class SslUtils:
    @staticmethod
    def create_ssl_adapter(verify_switch_env:str, ssl_cert_env:str, trigger_value: list):
        """设置SSL适配器，仅在启用SSL校验时挂载"""
        ssl_verify, ssl_cert = SslUtils.get_ssl_config(verify_switch_env, ssl_cert_env, trigger_value)
        if ssl_verify:
            class SSLAdapter(HTTPAdapter):
                def __init__(self, ssl_context, *args, **kwargs):
                    self.ssl_context = ssl_context
                    super().__init__(*args, **kwargs)

                def init_poolmanager(self, *args, **kwargs):
                    kwargs["ssl_context"] = self.ssl_context
                    return super().init_poolmanager(*args, **kwargs)

            ssl_context = SslUtils.create_strict_ssl_context(ssl_cert)
            adapter = SSLAdapter(ssl_context)
            return adapter

    @staticmethod
    def get_ssl_config(verify_switch_env:str, ssl_cert_env:str, trigger_value: list):
        """get ssl config"""
        is_ssl_verify_off = SslUtils._bool_env(verify_switch_env, trigger_value)
        ssl_cert = os.getenv(ssl_cert_env)

        if is_ssl_verify_off:
            return False, False

        if ssl_cert is None:
            raise ValueError(f"If verify_switch=true, must provide ssl_cert certificate")

        return True, ssl_cert

    @staticmethod
    def create_strict_ssl_context(ssl_cert: str = None) -> ssl.SSLContext:
        """创建严格的SSL上下文，要求TLS 1.2以上和指定的密码套件"""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1 | ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
        ctx.options |= ssl.OP_NO_RENEGOTIATION

        ctx.minimum_version = ssl.TLSVersion.TLSv1_2

        ctx.set_ciphers(
            "ECDHE-ECDSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-ECDSA-AES128-GCM-SHA256:"
            "ECDHE-RSA-AES128-GCM-SHA256"
        )

        if ssl_cert:
            if os.path.isfile(ssl_cert):
                abs_cert_path = os.path.abspath(ssl_cert)
                real_cert_path = os.path.realpath(ssl_cert)

                if abs_cert_path != real_cert_path:
                    raise ValueError(f"Certificate path contains symbolic links or path traversal attack.")

                if ".." in ssl_cert or ssl_cert.startswith("/") or "\\" in ssl_cert:
                    raise ValueError(f"The certificate path contains unsafe characters.")

                ctx.load_verify_locations(ssl_cert)

        return ctx

    @staticmethod
    def _bool_env(name: str, trigger_value: list) -> bool:
        """解析布尔型环境变量"""
        return os.getenv(name, "").strip().lower() in trigger_value
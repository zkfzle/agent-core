#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import ipaddress
import os
import re
import socket
from struct import unpack
from socket import inet_aton
from typing import Optional
from urllib.parse import urlparse

from openjiuwen.core.common.exception.status_code import StatusCode
from openjiuwen.core.common.security.exception_utils import ExceptionUtils


class UrlUtils:
    @staticmethod
    def check_url_is_valid(url):
        """check url is valid"""
        if not url:
            ExceptionUtils.raise_exception(StatusCode.URL_INVALID_ERROR, 'url is empty')
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if not re.match(r"^https?://.*$", url):
            ExceptionUtils.raise_exception(StatusCode.URL_INVALID_ERROR, 'illegal url protocol')
        try:
            ip_address = socket.gethostbyname(hostname)
        except socket.error:
            ExceptionUtils.raise_exception(StatusCode.URL_INVALID_ERROR, f"resolving IP address failed")
        if UrlUtils._is_inner_ipaddress(ip_address):
            ExceptionUtils.raise_exception(StatusCode.URL_INVALID_ERROR, f"illegal ip address")

    @staticmethod
    def get_global_proxy_url(url: str) -> Optional[str]:
        """get global proxy url"""
        no_proxy_list = UrlUtils._get_no_proxy_list()
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        global_proxy_url = None

        if not UrlUtils._is_no_proxy_match(hostname, no_proxy_list):
            global_proxy_url = os.getenv("http_proxy")

        if global_proxy_url:
            return global_proxy_url.strip()
        return global_proxy_url

    @staticmethod
    def get_global_proxies(url: str) -> Optional[dict]:
        """get global proxies"""
        global_proxy_url = UrlUtils.get_global_proxy_url(url)
        if global_proxy_url:
            return {
                "http": global_proxy_url,
                "https": global_proxy_url,
            }
        return None

    @staticmethod
    def _is_inner_ipaddress(ip):
        """judge inner ip"""
        if os.getenv("SSRF_PROTECT_ENABLED", "true").lower() == "false":
            # only if set SSRF_PROTECT_ENABLED to false, then allow inner ip
            return False

        ip_long = UrlUtils._ip_to_long(ip)
        is_inner_ip = UrlUtils._ip_to_long("10.0.0.0") <= ip_long <= UrlUtils._ip_to_long("10.255.255.255") or \
                      UrlUtils._ip_to_long("172.16.0.0") <= ip_long <= UrlUtils._ip_to_long("172.31.255.255") or \
                      UrlUtils._ip_to_long("192.168.0.0") <= ip_long <= UrlUtils._ip_to_long("192.168.255.255") or \
                      UrlUtils._ip_to_long("127.0.0.0") <= ip_long <= UrlUtils._ip_to_long("127.255.255.255") or \
                      ip_long == UrlUtils._ip_to_long("0.0.0.0")
        return is_inner_ip

    @staticmethod
    def _ip_to_long(ip_addr):
        """ trans ip to long"""
        return unpack("!L", inet_aton(ip_addr))[0]

    @staticmethod
    def _get_no_proxy_list() -> list[str]:
        no_proxy = os.getenv("NO_PROXY", "")
        no_proxy_list = [domain.strip() for domain in no_proxy.split(",") if domain.strip()]
        return no_proxy_list

    @staticmethod
    def _is_no_proxy_match(hostname: str, no_proxy_match: list[str]) -> bool:
        """check no proxy matchs"""
        if not no_proxy_match or not hostname:
            return False
        clean_hostname = hostname.strip().lower()
        for domain in no_proxy_match:
            clean_domain = domain.strip().lower()
            if not clean_domain:
                continue
            if clean_hostname.endswith(clean_domain):
                return True
        return False

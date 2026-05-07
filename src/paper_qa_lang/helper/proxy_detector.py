"""System proxy detection for paper-qa-lang.

Detects HTTP/HTTPS proxy configuration from environment variables
and Windows system registry settings.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass
class ProxyInfo:
    """Detected proxy configuration."""

    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None
    source: str = ""

    @property
    def enabled(self) -> bool:
        """Whether any proxy is configured."""
        return self.http_proxy is not None or self.https_proxy is not None

    @property
    def all(self) -> dict[str, str | None]:
        """All proxy settings as a dict (suitable for ``proxies=`` kwarg)."""
        return {
            "http": self.http_proxy,
            "https": self.https_proxy,
            "no_proxy": self.no_proxy,
        }


class ProxyDetector:
    """Detect system proxy configuration.

    Checks the following sources in order (first found wins):
    1. Environment variables ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY``
       (lowercase variants also checked).
    2. Windows registry (``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings``).

    Usage::

        info = ProxyDetector.detect()
        if info.enabled:
            requests.get(url, proxies=info.all)
    """

    @classmethod
    def detect(cls) -> ProxyInfo:
        """Detect proxy configuration from all available sources."""
        info = cls._from_env()
        if info.enabled:
            return info

        if sys.platform == "win32":
            info = cls._from_windows_registry()
            if info.enabled:
                return info

        return ProxyInfo(source="none")

    @classmethod
    def _from_env(cls) -> ProxyInfo:
        """Detect proxy from environment variables."""
        http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")

        if http or https:
            return ProxyInfo(
                http_proxy=http or None,
                https_proxy=https or None,
                no_proxy=no_proxy or None,
                source="env",
            )
        return ProxyInfo(source="env")

    @classmethod
    def _from_windows_registry(cls) -> ProxyInfo:
        """Detect proxy from Windows Internet Settings registry key."""
        try:
            import winreg  # noqa: PLC0415
        except ImportError:
            return ProxyInfo(source="winreg")

        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
            if not enabled:
                winreg.CloseKey(key)
                return ProxyInfo(source="winreg")

            server = winreg.QueryValueEx(key, "ProxyServer")[0]
            override = winreg.QueryValueEx(key, "ProxyOverride")[0]
            winreg.CloseKey(key)
        except OSError:
            return ProxyInfo(source="winreg")

        proxy_url = cls._normalize_proxy_server(server)
        no_proxy = override or None
        return ProxyInfo(
            http_proxy=proxy_url,
            https_proxy=proxy_url,
            no_proxy=no_proxy,
            source="winreg",
        )

    @staticmethod
    def _normalize_proxy_server(server: str) -> str:
        """Ensure proxy server string has a scheme."""
        server = server.strip()
        if not server.startswith(("http://", "https://", "socks://", "socks4://", "socks5://")):
            server = f"http://{server}"
        return server

    @classmethod
    def env_proxies(cls) -> dict[str, str]:
        """Build a ``proxies`` dict from env vars for ``httpx`` / ``requests``.

        Unlike :meth:`detect`, this only reads environment variables (matching
        the behaviour of ``httpx.Client(proxy=env_proxies())`` or
        ``requests.Session`` auto-detection) — it does **not** consult the
        Windows registry.
        """
        info = cls._from_env()
        return {k: v for k, v in info.all.items() if v is not None}

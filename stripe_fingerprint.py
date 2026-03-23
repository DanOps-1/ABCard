"""
Stripe 设备指纹 - guid / muid / sid 获取
通过请求 m.stripe.com/6 获取真实风控参数
"""
import json
import logging
import os
import re
import uuid
from typing import Optional

from http_client import create_http_session, USER_AGENT

logger = logging.getLogger(__name__)


class StripeFingerprint:
    """Stripe 设备指纹管理"""

    def __init__(self, proxy: Optional[str] = None, cache_key: str = "default"):
        self.session = create_http_session(proxy=proxy)
        self.proxy = proxy or ""
        self.cache_key = cache_key or "default"
        self._sticky = os.getenv("STRIPE_FINGERPRINT_STICKY", "1") not in ("0", "false", "False")
        self._cache_file = os.getenv("STRIPE_FINGERPRINT_CACHE_FILE", "outputs/stripe_fingerprint_cache.json")
        self.guid: str = ""
        self.muid: str = str(uuid.uuid4())  # __stripe_mid
        self.sid: str = str(uuid.uuid4())    # __stripe_sid

    def _cache_bucket(self) -> str:
        proxy_tag = self.proxy or "direct"
        return f"{self.cache_key}||{proxy_tag}"

    def _load_cache(self) -> bool:
        if not self._sticky:
            return False
        try:
            if not os.path.exists(self._cache_file):
                return False
            data = json.load(open(self._cache_file, "r", encoding="utf-8")) or {}
            item = data.get(self._cache_bucket()) or {}
            guid = (item.get("guid") or "").strip()
            muid = (item.get("muid") or "").strip()
            sid = (item.get("sid") or "").strip()
            if not (guid and muid and sid):
                return False
            self.guid, self.muid, self.sid = guid, muid, sid
            logger.info("复用 Stripe 指纹缓存: guid=%s..., key=%s", self.guid[:12], self._cache_bucket()[:64])
            return True
        except Exception as e:
            logger.debug("读取 Stripe 指纹缓存失败: %s", e)
            return False

    def _save_cache(self):
        if not self._sticky:
            return
        try:
            os.makedirs(os.path.dirname(self._cache_file) or ".", exist_ok=True)
            data = {}
            if os.path.exists(self._cache_file):
                try:
                    data = json.load(open(self._cache_file, "r", encoding="utf-8")) or {}
                except Exception:
                    data = {}
            data[self._cache_bucket()] = {
                "guid": self.guid,
                "muid": self.muid,
                "sid": self.sid,
            }
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("保存 Stripe 指纹缓存失败: %s", e)

    def fetch_from_m_stripe(self) -> bool:
        """
        从 m.stripe.com/6 获取 guid/muid/sid。
        这是 Stripe 的设备指纹采集端点。
        """
        if self._load_cache():
            return True

        logger.info("获取 Stripe 设备指纹 (m.stripe.com/6)...")
        try:
            headers = {
                "Accept": "*/*",
                "Origin": "https://m.stripe.network",
                "Referer": "https://m.stripe.network/",
                "User-Agent": USER_AGENT,
            }
            resp = self.session.post(
                "https://m.stripe.com/6",
                headers=headers,
                json={
                    "v": "m-outer-3437aaddcdf6922d623e172c2d6f9278",
                    "t": 0,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json() if hasattr(resp, 'json') else json.loads(resp.text)
                # 从返回数据中提取指纹
                self.guid = data.get("guid", data.get("id", ""))
                if data.get("muid"):
                    self.muid = data["muid"]
                if data.get("sid"):
                    self.sid = data["sid"]
                logger.info(f"Stripe 指纹获取成功 - guid: {self.guid[:12]}...")
                self._save_cache()
                return True
            else:
                logger.warning(f"m.stripe.com/6 返回 {resp.status_code}, 使用模拟值")
        except Exception as e:
            logger.warning(f"获取 Stripe 指纹失败: {e}, 使用模拟值")

        # fallback: 生成模拟值
        if not self.guid:
            self.guid = str(uuid.uuid4())
            logger.info("使用模拟 guid (注意：可能触发 3DS 验证)")
        self._save_cache()

        return False

    def get_params(self) -> dict:
        """返回用于 confirm 请求的指纹参数"""
        return {
            "guid": self.guid,
            "muid": self.muid,
            "sid": self.sid,
        }

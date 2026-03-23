"""
支付流程 - Checkout + Confirm
主链路:
  1. POST /backend-api/payments/checkout  -> checkout_session_id + publishable_key
  2. 获取 Stripe 指纹 (guid/muid/sid)
  3. POST /v1/payment_methods -> 卡片 tokenization
  4. POST /v1/payment_pages/{checkout_session_id}/confirm -> 支付确认
  5. 如触发 Stripe hCaptcha 挑战 (intent_confirmation_challenge)，按 CTF 规则直接判负
"""
import json
import logging
import os
import re
import time
import uuid
import ipaddress
from urllib.parse import urlsplit, urlunsplit, urlencode
from typing import Optional

from config import Config, CardInfo, BillingInfo
from auth_flow import AuthResult
from stripe_fingerprint import StripeFingerprint
from captcha_solver import CaptchaSolver
from http_client import create_http_session, USER_AGENT
from sentinel import get_sentinel_token

logger = logging.getLogger(__name__)


class PaymentResult:
    """支付结果"""

    def __init__(self):
        self.checkout_session_id: str = ""
        self.confirm_status: str = ""
        self.confirm_response: dict = {}
        self.confirm_initial_response: dict = {}
        self.verify_response: dict = {}
        self.challenge_context: dict = {}
        self.checkout_data: dict = {}  # ChatGPT checkout 原始返回
        self.openai_checkout_url: str = ""
        self.openai_client_secret: str = ""
        self.experiment_tag: dict = {}
        self.success: bool = False
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "checkout_session_id": self.checkout_session_id,
            "confirm_status": self.confirm_status,
            "success": self.success,
            "error": self.error,
            "confirm_response": self.confirm_response,
            "confirm_initial_response": self.confirm_initial_response,
            "verify_response": self.verify_response,
            "challenge_context": self.challenge_context,
            "checkout_data": self.checkout_data,
            "openai_checkout_url": self.openai_checkout_url,
            "openai_client_secret": self.openai_client_secret,
            "experiment_tag": self.experiment_tag,
        }


class PaymentFlow:
    """支付协议流"""

    def __init__(self, config: Config, auth_result: AuthResult, stripe_proxy: str = None):
        self.config = config
        self.auth = auth_result
        self.session = create_http_session(proxy=config.proxy)  # ChatGPT 用 proxy
        # Stripe 调用的代理 (None=直连, 或设为与 ChatGPT 同代理实现 IP 一致性)
        self._stripe_proxy = stripe_proxy
        self._stripe_session = create_http_session(proxy=stripe_proxy)
        fp_cache_key = (getattr(auth_result, "email", "") or "anon").strip().lower() or "anon"
        self.fingerprint = StripeFingerprint(proxy=stripe_proxy, cache_key=fp_cache_key)
        self.result = PaymentResult()
        self.stripe_pk: str = ""  # Stripe publishable key
        self.checkout_url: str = ""  # Stripe checkout URL
        self.checkout_data: dict = {}  # 完整 checkout 响应
        self.payment_method_id: str = ""  # tokenized payment method ID
        self._risk_strict_mode: bool = os.getenv("RISK_STRICT_MODE", "1") not in ("0", "false", "False")
        self._skip_checkout_page_fetch: bool = os.getenv("SKIP_CHECKOUT_PAGE_FETCH", "1") not in ("0", "false", "False")
        # 赛题默认需要执行最终确认阶段，因此默认关闭 link-only
        self._openai_link_only: bool = os.getenv("OPENAI_LINK_ONLY", "0") not in ("0", "false", "False")
        self._strict_confirm: bool = os.getenv("STRICT_CONFIRM", "1") not in ("0", "false", "False")
        # INIT 命中 hCaptcha 风险信号时，是否提前短路判负（默认关闭，继续跑到 confirm）
        self._init_hcaptcha_auto_fail: bool = os.getenv("INIT_HCAPTCHA_AUTO_FAIL", "0") not in ("0", "false", "False")
        # 是否在 confirm 命中 intent_confirmation_challenge 后自动尝试打码
        self._enable_hcaptcha_solver: bool = os.getenv("ENABLE_HCAPTCHA_SOLVER", "1") not in ("0", "false", "False")
        # 可选：先尝试浏览器 handleNextAction（默认开启；失败会回退到打码）
        self._enable_browser_challenge: bool = os.getenv("ENABLE_BROWSER_CHALLENGE", "1") not in ("0", "false", "False")
        # 打码最大轮数（Stripe 可能返回多轮 challenge）
        try:
            self._hcaptcha_max_rounds: int = max(1, int(os.getenv("HCAPTCHA_MAX_ROUNDS", "2")))
        except Exception:
            self._hcaptcha_max_rounds = 2
        self._exp_run_id: str = uuid.uuid4().hex[:12]
        self._exp_promo_variant: str = os.getenv("PROMO_VARIANT", "A").strip().upper() or "A"
        self._exp_promo_id: str = os.getenv("PROMO_ID", "").strip()

        # 设置认证 cookie
        self.session.cookies.set(
            "__Secure-next-auth.session-token",
            auth_result.session_token,
            domain=".chatgpt.com",
        )
        if auth_result.device_id:
            self.session.cookies.set("oai-did", auth_result.device_id, domain=".chatgpt.com")

    def _get_sentinel_token(self) -> str:
        """获取支付场景的 sentinel token（完整 PoW 版）"""
        if not self.auth.device_id:
            self.auth.device_id = str(uuid.uuid4())
        return get_sentinel_token(
            self.session,
            device_id=self.auth.device_id,
            flow="authorize_continue",
        )

    # ── Step 1: 创建 Checkout Session ──
    def _create_checkout_via_aimizy(self) -> dict:
        """通过 aimizy 生成带优惠的支付链接（完整参数匹配 HAR）"""
        plan = self.config.team_plan
        billing = self.config.billing
        body = {
            "access_token": self.auth.access_token,
            "plan_name": plan.plan_name,
            "country": billing.country,
            "currency": billing.currency,
            "promo_campaign_id": plan.promo_campaign_id,
            "is_coupon_from_query_param": True,
            "seat_quantity": plan.seat_quantity,
            "price_interval": plan.price_interval,
            "check_card_proxy": False,
            "is_short_link": True,
        }
        aimizy_session = create_http_session(proxy=self.config.proxy)
        resp = aimizy_session.post(
            "https://team.aimizy.com/api/public/generate-payment-link",
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://team.aimizy.com",
                "Referer": "https://team.aimizy.com/pay",
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"aimizy 请求失败: {resp.status_code} - {resp.text[:200]}")
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"aimizy 返回失败: {data}")
        return data

    def create_checkout_session(self) -> str:
        """
        创建 Checkout Session
        主路径: ChatGPT checkout API + sentinel token (不触发 hCaptcha)
        Fallback: aimizy (可能触发 hCaptcha)
        """
        logger.info("[支付 1/3] 创建 Checkout Session...")
        logger.info(
            "checkout 配置参数: plan=%s country=%s currency=%s seats=%s interval=%s promo=%s",
            self.config.team_plan.plan_name,
            self.config.billing.country,
            self.config.billing.currency,
            self.config.team_plan.seat_quantity,
            self.config.team_plan.price_interval,
            self.config.team_plan.promo_campaign_id,
        )
        if self._risk_strict_mode and self.config.proxy != self._stripe_proxy:
            logger.warning(
                "RISK_STRICT_MODE: ChatGPT proxy 与 Stripe proxy 不一致 (chatgpt=%s, stripe=%s)",
                self.config.proxy,
                self._stripe_proxy,
            )
        if self._risk_strict_mode and self.config.billing.country != "US":
            logger.warning("RISK_STRICT_MODE: 当前 billing.country=%s，可能增加风控", self.config.billing.country)

        plan = self.config.team_plan
        billing = self.config.billing

        # ── 主路径: ChatGPT checkout API + sentinel token ──
        sentinel = self._get_sentinel_token()
        device_id = self.auth.device_id

        headers = {
            "Authorization": f"Bearer {self.auth.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Origin": "https://chatgpt.com",
            "Referer": "https://chatgpt.com/",
            "oai-device-id": device_id,
            "openai-sentinel-token": sentinel,
            "User-Agent": USER_AGENT,
        }

        promo_id = self._exp_promo_id or plan.promo_campaign_id
        variant = self._exp_promo_variant if self._exp_promo_variant in ("A", "B") else "A"

        if promo_id in ("none", "null", "off", "-"):
            promo_id = ""

        body = {
            "plan_name": plan.plan_name,
            "billing_details": {
                "country": billing.country,
                "currency": billing.currency,
            },
            "checkout_ui_mode": "custom",
            "team_plan_data": {
                "workspace_name": plan.workspace_name,
                "price_interval": plan.price_interval,
                "seat_quantity": plan.seat_quantity,
            },
            "promo_campaign": {
                "promo_campaign_id": promo_id,
                "is_coupon_from_query_param": True,
            },
        }
        if not promo_id:
            body.pop("promo_campaign", None)

        if variant == "B":
            body["team_plan_data"]["price_interval"] = "month"
            body["team_plan_data"]["seat_quantity"] = 2
            if promo_id:
                body["promo_campaign_id"] = promo_id
                body["is_coupon_from_query_param"] = True

        exp_tag = {
            "mode": "ctf",
            "run_id": self._exp_run_id,
            "promo_variant": variant,
            "promo_id": promo_id,
            "seat_quantity": body.get("team_plan_data", {}).get("seat_quantity"),
            "price_interval": body.get("team_plan_data", {}).get("price_interval"),
            "stage": "checkout",
        }
        self.result.experiment_tag = {
            "mode": "ctf",
            "run_id": self._exp_run_id,
            "promo_variant": variant,
            "promo_id": promo_id,
            "seat_quantity": body.get("team_plan_data", {}).get("seat_quantity"),
            "price_interval": body.get("team_plan_data", {}).get("price_interval"),
        }

        logger.info("exp_tag=%s", json.dumps(exp_tag, ensure_ascii=False))
        logger.info(
            "checkout 实际发包参数: variant=%s seats=%s interval=%s promo=%s has_promo_obj=%s has_promo_top_level=%s",
            variant,
            body.get("team_plan_data", {}).get("seat_quantity"),
            body.get("team_plan_data", {}).get("price_interval"),
            promo_id,
            bool(body.get("promo_campaign")),
            bool(body.get("promo_campaign_id")),
        )

        resp = self.session.post(
            "https://chatgpt.com/backend-api/payments/checkout",
            headers=headers,
            json=body,
            timeout=30,
        )
        if resp.status_code != 200:
            err = f"{resp.status_code} {resp.text[:200]}"
            logger.warning("checkout 单变体失败: %s", err)
            logger.info("Fallback 到 aimizy...")
            return self._create_checkout_via_aimizy_fallback()

        data = resp.json()
        sd = data.get("scheduled_discount_preview")
        im = data.get("immediate_discount_settings")
        logger.info(
            "checkout 单变体返回: scheduled_discount=%s immediate_discount=%s",
            bool(sd),
            bool(im),
        )
        logger.info(f"Checkout 返回字段: {list(data.keys())}")
        logger.debug(f"Checkout 返回内容: {json.dumps(data, ensure_ascii=False)[:2000]}")
        if data.get("client_secret"):
            logger.info(f"client_secret: {data['client_secret'][:40]}...")

        # 保存 OpenAI checkout 字段（赛题格式）
        processor_entity = data.get("processor_entity", "openai_llc")
        self.result.openai_client_secret = data.get("client_secret", "") or ""

        # 保存 checkout_url 和 publishable_key
        self.checkout_url = data.get("url", "") or data.get("checkout_url", "") or ""
        pk_from_response = data.get("publishable_key", "")
        if pk_from_response:
            self.stripe_pk = pk_from_response
            logger.info(f"Stripe PK (from checkout): {self.stripe_pk[:30]}...")

        # 保存完整 checkout 返回数据
        self.checkout_data = data
        self.result.checkout_data = data

        # 记录 promo/discount 命中情况
        sd = data.get("scheduled_discount_preview")
        im = data.get("immediate_discount_settings")
        provider = data.get("checkout_provider", "")
        logger.info("checkout 诊断: provider=%s scheduled_discount=%s immediate_discount=%s", provider, bool(sd), bool(im))
        logger.info(
            "exp_tag=%s",
            json.dumps(
                {
                    **self.result.experiment_tag,
                    "stage": "checkout_result",
                    "outcome": "discount_hit" if (sd or im) else "discount_miss",
                    "provider": provider,
                },
                ensure_ascii=False,
            ),
        )
        if sd or im:
            logger.info(f"Promo 命中! scheduled={sd}, immediate={im}")
        elif promo_id:
            logger.warning(f"Promo 未命中(主链路): id={promo_id}")

        # 从返回提取 checkout_session_id
        cs_id = (
            data.get("checkout_session_id")
            or data.get("session_id")
            or ""
        )

        # 从 checkout_url 中提取
        if not cs_id:
            checkout_url = self.checkout_url
            if "cs_" in checkout_url:
                m = re.search(r"(cs_[A-Za-z0-9_]+)", checkout_url)
                if m:
                    cs_id = m.group(1)

        # 从 client_secret 中提取
        if not cs_id:
            secret = data.get("client_secret", "")
            if secret and "_secret_" in secret:
                cs_id = secret.split("_secret_")[0]

        if not cs_id:
            raise RuntimeError(f"未能从返回中提取 checkout_session_id: {data}")

        self.result.checkout_session_id = cs_id
        self.result.openai_checkout_url = f"https://chatgpt.com/checkout/{processor_entity}/{cs_id}"
        logger.info(f"Checkout Session ID: {cs_id[:30]}...")
        logger.info(f"OpenAI 订阅链接: {self.result.openai_checkout_url}")
        return cs_id

    def _create_checkout_via_aimizy_fallback(self) -> str:
        """Fallback: 通过 aimizy 获取带优惠的 checkout session"""
        aimizy_data = self._create_checkout_via_aimizy()
        checkout_url = aimizy_data.get("url", "")
        cs_id = aimizy_data.get("checkout_session_id", "")
        if not cs_id and "cs_" in checkout_url:
            m = re.search(r"(cs_[A-Za-z0-9_]+)", checkout_url)
            if m:
                cs_id = m.group(1)
        if not cs_id:
            raise RuntimeError(f"aimizy fallback 未能提取 cs_id: {aimizy_data}")
        logger.info(f"aimizy fallback 获取成功: cs_id={cs_id[:40]}...")
        logger.warning("当前链路: fallback(aimizy)")
        self.checkout_url = checkout_url
        self.result.checkout_session_id = cs_id
        self.checkout_data = aimizy_data
        self.result.checkout_data = aimizy_data
        return cs_id

    # ── Step 2: 获取 Stripe 指纹 ──
    def fetch_stripe_fingerprint(self):
        """获取 guid/muid/sid"""
        logger.info("[支付 2/4] 获取 Stripe 设备指纹...")
        self.fingerprint.fetch_from_m_stripe()

    # 已知的 OpenAI Stripe pk_live (公开嵌入在 pay.openai.com 页面中)
    OPENAI_STRIPE_PK_LIVE = "pk_live_51LGShDDngiIFhMSFKJORJiGPcesLuJwg3TOFRDqz3bEuQxilcMq5RJFCm0XxzmDnGlJ6GtsfVOmGPMxJOBIM7kde00kMOxMPFI"

    # ── Step 2.5: 提取 Stripe publishable key ──
    def extract_stripe_pk(self, checkout_url: str) -> str:
        """
        从 checkout 页面或 payment_pages 接口提取 Stripe publishable key.
        优先 pk_live_，fallback 到已知 OpenAI pk。
        """
        logger.info("[支付 3/4] 获取 Stripe Publishable Key...")

        # 如果已经从 checkout 响应中获取到了，直接返回
        if self.stripe_pk and self.stripe_pk.startswith("pk_live_"):
            logger.info(f"已有 Stripe PK (live): {self.stripe_pk[:30]}...")
            return self.stripe_pk

        cs_id = self.result.checkout_session_id

        # 如果没有 checkout_url，尝试构造
        if not checkout_url and cs_id:
            checkout_url = f"https://checkout.stripe.com/c/pay/{cs_id}"

        # 比赛模式：已从 checkout 响应拿到 live key 时，避免额外页面访问增加噪声
        if self._skip_checkout_page_fetch and self.stripe_pk and self.stripe_pk.startswith("pk_live_"):
            return self.stripe_pk

        # 方法 1: 从 checkout 页面提取 (优先 pk_live_)
        if checkout_url:
            try:
                headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "User-Agent": USER_AGENT,
                }
                resp = self.session.get(checkout_url, headers=headers, timeout=30, allow_redirects=True)
                logger.debug(f"Checkout 页面状态: {resp.status_code}, 长度: {len(resp.text)}")
                if resp.status_code == 200:
                    # 优先匹配 pk_live_
                    live_match = re.search(r'(pk_live_[A-Za-z0-9]+)', resp.text)
                    if live_match:
                        self.stripe_pk = live_match.group(1)
                        logger.info(f"Stripe PK (live): {self.stripe_pk[:30]}...")
                        return self.stripe_pk
                    # 其次匹配 pk_test_ (仅作日志记录，不使用)
                    test_match = re.search(r'(pk_test_[A-Za-z0-9]+)', resp.text)
                    if test_match:
                        logger.warning(f"页面仅找到 test key: {test_match.group(1)[:30]}... (跳过，使用 fallback)")
                    else:
                        logger.debug("checkout 页面中未找到 pk_ 模式")
            except Exception as e:
                logger.warning(f"从 checkout 页面提取 PK 失败: {e}")

        # 方法 2: 从 payment_pages/{cs_id} 获取
        if cs_id:
            try:
                stripe_session = self._stripe_session
                resp = stripe_session.get(
                    f"https://api.stripe.com/v1/payment_pages/{cs_id}",
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
                logger.debug(f"payment_pages 状态: {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    pk = data.get("merchant", {}).get("publishable_key", "")
                    if not pk:
                        pk = data.get("publishable_key", "")
                    if pk and pk.startswith("pk_live_"):
                        self.stripe_pk = pk
                        logger.info(f"Stripe PK (from payment_pages): {self.stripe_pk[:30]}...")
                        return self.stripe_pk
                    elif pk:
                        logger.warning(f"payment_pages 返回非 live key: {pk[:30]}...")
                    else:
                        logger.debug(f"payment_pages 返回字段: {list(data.keys())}")
            except Exception as e:
                logger.warning(f"从 payment_pages 提取 PK 失败: {e}")

        # 方法 3: 硬编码的已知 OpenAI pk_live fallback
        logger.info(f"使用已知 OpenAI pk_live fallback: {self.OPENAI_STRIPE_PK_LIVE[:30]}...")
        self.stripe_pk = self.OPENAI_STRIPE_PK_LIVE
        return self.stripe_pk

    # ── Step 3: 创建支付方式 (卡片 tokenization) ──
    def create_payment_method(self) -> str:
        """
        POST /v1/payment_methods
        先将卡片信息 tokenize, 返回 pm_xxx ID
        Stripe 限制直接在 confirm 中提交原始卡号
        """
        logger.info("[支付 3.5/5] 创建 Payment Method (卡片 tokenization)...")

        card = self.config.card
        billing = self.config.billing
        fp = self.fingerprint.get_params()

        form_data = {
            "type": "card",
            "card[number]": card.number,
            "card[cvc]": card.cvc,
            "card[exp_month]": card.exp_month,
            "card[exp_year]": card.exp_year,
            "billing_details[name]": billing.name,
            "billing_details[email]": billing.email or self.auth.email,
            "billing_details[address][country]": billing.country,
            "billing_details[address][line1]": billing.address_line1,
            "billing_details[address][line2]": getattr(billing, "address_line2", "") or "",
            "billing_details[address][city]": getattr(billing, "address_city", "") or "",
            "billing_details[address][state]": billing.address_state,
            "billing_details[address][postal_code]": billing.postal_code,
            "allow_redisplay": "always",
            "guid": fp["guid"],
            "muid": fp["muid"],
            "sid": fp["sid"],
            "payment_user_agent": f"stripe.js/{self.config.stripe_build_hash}; stripe-js-v3/{self.config.stripe_build_hash}; checkout",
        }

        headers = {
            "Authorization": f"Bearer {self.stripe_pk}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Origin": "https://js.stripe.com",
            "Referer": "https://js.stripe.com/",
            "User-Agent": USER_AGENT,
        }

        # Stripe API 使用配置的代理 (可能直连或走代理)
        stripe_session = self._stripe_session
        resp = stripe_session.post(
            "https://api.stripe.com/v1/payment_methods",
            headers=headers,
            data=form_data,
            timeout=30,
        )

        if resp.status_code != 200:
            # 保存原始 Stripe 响应供 UI 展示
            try:
                self.result.confirm_response = resp.json()
            except Exception:
                self.result.confirm_response = {"raw": resp.text[:500]}
            self.result.confirm_status = str(resp.status_code)

            err = resp.text[:300]
            try:
                err = resp.json().get("error", {}).get("message", err)
            except Exception:
                pass
            raise RuntimeError(f"创建 Payment Method 失败 ({resp.status_code}): {err}")

        pm_data = resp.json()
        pm_id = pm_data.get("id", "")
        logger.info(f"Payment Method ID: {pm_id[:20]}...")
        return pm_id

    # 数字商品 VAT/GST 税率表 (用于 automatic_tax 场景下计算 expected_amount)
    COUNTRY_TAX_RATES = {
        "US": 0.00,     # 大部分州数字商品免税 (但有例外)
        "GB": 0.20,     # UK VAT 20%
        "DE": 0.19,     # Germany 19%
        "FR": 0.20,     # France 20%
        "JP": 0.10,     # Japan 10%
        "SG": 0.09,     # Singapore GST 9%
        "HK": 0.00,     # Hong Kong 0%
        "KR": 0.10,     # Korea 10%
        "AU": 0.10,     # Australia GST 10%
        "CA": 0.05,     # Canada GST 5% (最低, HST varies)
        "NL": 0.21,     # Netherlands 21%
        "IT": 0.22,     # Italy 22%
        "ES": 0.21,     # Spain 21%
        "CH": 0.081,    # Switzerland 8.1%
        "IE": 0.23,     # Ireland 23%
        "SE": 0.25,     # Sweden 25%
        "NO": 0.25,     # Norway 25%
        "DK": 0.25,     # Denmark 25%
        "BE": 0.21,     # Belgium 21%
        "AT": 0.20,     # Austria 20%
        "PT": 0.23,     # Portugal 23%
        "FI": 0.255,    # Finland 25.5%
        "PL": 0.23,     # Poland 23%
        "CZ": 0.21,     # Czech Republic 21%
    }

    # ── Step 3.7: 初始化支付页面 + 获取 expected_amount ──
    def fetch_payment_page_details(self, checkout_session_id: str) -> int:
        """
        初始化 Stripe 支付页面并获取 expected_amount (含税):
        1) POST /v1/payment_pages/{cs_id}/init  → 获取 base amount, eid, init_checksum
        2) 根据 billing_country 的 automatic_tax 税率计算含税金额
        """
        logger.info("[支付 3.7/5] 初始化支付页面 & 获取 expected_amount...")

        stripe_session = self._stripe_session

        headers_form = {
            "Authorization": f"Bearer {self.stripe_pk}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Origin": "https://js.stripe.com",
            "Referer": "https://js.stripe.com/",
            "User-Agent": USER_AGENT,
        }

        # 1) payment_pages/{cs}/init
        init_url = f"https://api.stripe.com/v1/payment_pages/{checkout_session_id}/init"
        init_form = {
            "key": self.stripe_pk,
            "browser_locale": "en",
        }
        init_resp = stripe_session.post(init_url, headers=headers_form, data=init_form, timeout=30)
        if init_resp.status_code != 200:
            logger.warning(f"payment_pages/init 失败 {init_resp.status_code}: {init_resp.text[:500]}")
            self._expected_amount = "0"
            self._init_eid = ""
            self._init_checksum = ""
            return 0

        init_data = init_resp.json()
        self._init_data = init_data

        # 保存 eid 和 init_checksum (confirm 时需要)
        self._init_eid = init_data.get("eid", "")
        self._init_checksum = init_data.get("init_checksum", "")
        # 保存 stripe_hosted_url (便于挑战上下文诊断)
        self._stripe_hosted_url = init_data.get("stripe_hosted_url", "")

        # 提取基础金额 (税前)
        total_summary = init_data.get("total_summary", {})
        base_amount = total_summary.get("due", 0)
        logger.info(f"init base amount: {base_amount} (total_summary.due)")
        try:
            recurring = ((init_data.get("line_items") or [{}])[0].get("price") or {}).get("unit_amount")
        except Exception:
            recurring = None
        logger.info(
            "exp_tag=%s",
            json.dumps(
                {
                    **self.result.experiment_tag,
                    "stage": "pricing_snapshot",
                    "init_due": base_amount,
                    "unit_amount": recurring,
                    "promo_preview_hit": bool(self.checkout_data.get("scheduled_discount_preview") or self.checkout_data.get("immediate_discount_settings")),
                },
                ensure_ascii=False,
            ),
        )

        # 检查是否需要计算税金
        tax_meta = init_data.get("tax_meta", {})
        auto_tax = init_data.get("tax_context", {}).get("automatic_tax_enabled", False)

        if auto_tax and tax_meta.get("status") == "requires_location_inputs":
            # 需要根据 billing country 的税率计算含税金额
            billing_country = self.config.billing.country
            tax_rate = self.COUNTRY_TAX_RATES.get(billing_country, 0.0)
            amount_with_tax = round(base_amount * (1 + tax_rate))
            logger.info(f"automatic_tax: country={billing_country}, rate={tax_rate*100:.1f}%, "
                        f"base={base_amount}, with_tax={amount_with_tax}")
            self._expected_amount = str(amount_with_tax)
            return amount_with_tax
        else:
            # 税已包含或不需要税
            logger.info(f"expected_amount (no tax adj): {base_amount}")
            self._expected_amount = str(base_amount) if base_amount else "0"
            return base_amount

    def _init_indicates_hcaptcha(self) -> bool:
        """基于 init 响应提前判定是否会触发 hCaptcha，命中则按 CTF 规则直接失败。"""
        data = getattr(self, "_init_data", {}) or {}

        pi = data.get("payment_intent") or {}
        si = data.get("setup_intent") or {}
        intent = pi or si
        next_action = intent.get("next_action") or {}
        sdk_info = next_action.get("use_stripe_sdk") or {}

        pi_status = pi.get("status", "")
        si_status = si.get("status", "")
        next_action_type = next_action.get("type", "")
        challenge_type = sdk_info.get("type", "")

        has_site_key = bool(data.get("site_key"))
        has_rqdata = bool(data.get("rqdata"))
        has_link_hcaptcha = bool((data.get("link_settings") or {}).get("hcaptcha_site_key"))
        feature_flags = data.get("feature_flags") or {}
        passive_captcha = bool(feature_flags.get("checkout_passive_captcha"))
        link_hcaptcha_rqdata = bool(feature_flags.get("checkout_enable_link_api_hcaptcha_rqdata"))

        # 记录 init 风险快照，便于 A/B 精准比对（避免重复刷屏）
        if not getattr(self, "_init_risk_logged", False):
            logger.info(
                "exp_tag=%s",
                json.dumps(
                    {
                        **self.result.experiment_tag,
                        "stage": "init_risk_snapshot",
                        "pi_status": pi_status,
                        "si_status": si_status,
                        "next_action_type": next_action_type,
                        "sdk_type": challenge_type,
                        "has_site_key": has_site_key,
                        "has_rqdata": has_rqdata,
                        "has_link_hcaptcha": has_link_hcaptcha,
                        "passive_captcha": passive_captcha,
                        "link_hcaptcha_rqdata": link_hcaptcha_rqdata,
                    },
                    ensure_ascii=False,
                ),
            )
            self._init_risk_logged = True

        # 高置信度命中条件：init 已明确包含挑战类型
        if challenge_type == "intent_confirmation_challenge":
            return True

        # 兼容不同返回结构：init JSON 中已出现挑战标记
        try:
            raw = json.dumps(data, ensure_ascii=False)
        except Exception:
            raw = ""
        if "intent_confirmation_challenge" in raw:
            return True

        # 中高风险命中：init 已带完整 captcha 上下文，提前短路避免无效 tokenization
        if has_site_key and has_rqdata and has_link_hcaptcha and (passive_captcha or link_hcaptcha_rqdata):
            return True

        return False

    # ── Step 4: 确认支付 ──
    def confirm_payment(self, checkout_session_id: str) -> PaymentResult:
        """
        POST /v1/payment_pages/{checkout_session_id}/confirm
        匹配浏览器原生 Stripe checkout 格式 (卡数据内联, 完整 metadata)
        """
        logger.info("[支付 4/5] 确认支付...")

        if self._init_indicates_hcaptcha():
            if self._init_hcaptcha_auto_fail:
                logger.info("init 已预判会触发 hCaptcha，INIT_HCAPTCHA_AUTO_FAIL=1，直接判负并跳过 confirm")
                self.result.confirm_status = "skipped_hcaptcha_predicted"
                self.result.error = "hcaptcha_predicted_from_init_auto_fail"
                logger.info(
                    "exp_tag=%s",
                    json.dumps(
                        {
                            **self.result.experiment_tag,
                            "stage": "challenge",
                            "outcome": "failed",
                            "reason": "hcaptcha_predicted_from_init_auto_fail",
                        },
                        ensure_ascii=False,
                    ),
                )
                return self.result
            logger.warning("init 预判命中 hCaptcha，但 INIT_HCAPTCHA_AUTO_FAIL=0，继续执行 confirm")

        expected = getattr(self, '_expected_amount', "0")
        checksum = getattr(self, '_init_checksum', "")
        card = self.config.card
        billing = self.config.billing
        fp = self.fingerprint.get_params()
        stripe_js_id = str(uuid.uuid4())
        elements_session_id = f"elements_session_{uuid.uuid4().hex[:12]}"

        # 统一为 payment_method confirm，避免与 tokenize 路径冲突
        form_data = {
            "guid": fp["guid"],
            "muid": fp["muid"],
            "sid": fp["sid"],
            "payment_method": self.payment_method_id,
            "expected_amount": expected,
            "expected_payment_method_type": "card",
            "elements_session_client[session_id]": elements_session_id,
            "elements_session_client[stripe_js_id]": stripe_js_id,
            "elements_session_client[locale]": "en",
            "elements_session_client[referrer_host]": "chatgpt.com",
            "client_attribution_metadata[client_session_id]": stripe_js_id,
            "client_attribution_metadata[checkout_session_id]": checkout_session_id,
            "client_attribution_metadata[merchant_integration_source]": "checkout",
            "client_attribution_metadata[merchant_integration_version]": "custom",
            "client_attribution_metadata[merchant_integration_subtype]": "payment-element",
            "key": self.stripe_pk,
            "_stripe_version": "2025-03-31.basil; checkout_server_update_beta=v1; checkout_manual_approval_preview=v1",
        }
        eid = getattr(self, '_init_eid', "")
        if eid:
            form_data["eid"] = eid
        if checksum:
            form_data["init_checksum"] = checksum

        logger.info(
            "confirm 参数: expected_amount=%s, pm=%s..., eid=%s..., has_checksum=%s, guid=%s..., muid=%s..., sid=%s...",
            expected,
            (self.payment_method_id[:18] if self.payment_method_id else ""),
            (eid[:12] if eid else ""),
            bool(checksum),
            (fp.get("guid", "")[:12]),
            (fp.get("muid", "")[:12]),
            (fp.get("sid", "")[:12]),
        )

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Origin": "https://js.stripe.com",
            "Referer": "https://js.stripe.com/",
            "User-Agent": USER_AGENT,
        }

        url = f"https://api.stripe.com/v1/payment_pages/{checkout_session_id}/confirm"
        stripe_session = self._stripe_session
        resp = stripe_session.post(url, headers=headers, data=form_data, timeout=60)

        self.result.confirm_status = str(resp.status_code)
        try:
            self.result.confirm_response = resp.json()
        except Exception:
            self.result.confirm_response = {"raw": resp.text[:500]}

        if resp.status_code == 200:
            data = resp.json()
            self.result.confirm_initial_response = data
            logger.debug(f"confirm 响应: {json.dumps(data, ensure_ascii=False)[:1000]}")
            status = data.get("status", "")
            pi = data.get("payment_intent") or {}
            si = data.get("setup_intent") or {}
            pi_status = pi.get("status", "") or si.get("status", "")
            next_action = pi.get("next_action", {}) or si.get("next_action", {})
            intent_id = pi.get("id", "") or si.get("id", "")
            next_action_type = (next_action or {}).get("type", "")
            sdk_type = ((next_action.get("use_stripe_sdk") or {}).get("type") if isinstance(next_action, dict) else "")
            logger.info(
                "confirm 结果: http=%s session=%s intent=%s intent_id=%s... next_action=%s sdk_type=%s",
                resp.status_code,
                status or "",
                pi_status or "",
                (intent_id[:20] if intent_id else ""),
                next_action_type or "",
                sdk_type or "",
            )

            if status == "complete" or pi_status == "succeeded":
                self.result.success = True
                logger.info("支付确认成功!")
            elif status == "open" and not pi and not si.get("next_action"):
                # amount=0 场景: confirm 后需要 poll 等待 session 完成
                logger.info("confirm 返回 open (amount=0), 轮询等待完成...")
                poll_result = self._poll_payment_page(checkout_session_id)
                if poll_result:
                    self.result.success = True
                    logger.info("支付确认成功 (poll 完成)!")
                else:
                    self.result.error = "poll 超时，session 未完成"
                    logger.error(self.result.error)
            elif pi_status == "requires_action" and next_action:
                # Stripe Radar challenge / 3DS
                sdk_info = next_action.get("use_stripe_sdk", {})
                challenge_type = sdk_info.get("type", "")
                intent_client_secret = pi.get("client_secret", "") or si.get("client_secret", "")

                if challenge_type == "intent_confirmation_challenge":
                    logger.info("Stripe 要求 hCaptcha 挑战验证 (intent_confirmation_challenge)")
                    stripe_js = sdk_info.get("stripe_js", {})
                    intent_id = pi.get("id", "") or si.get("id", "")
                    client_secret = pi.get("client_secret", "") or si.get("client_secret", "")
                    site_key = stripe_js.get("site_key", "")
                    rqdata = stripe_js.get("rqdata", "")
                    verification_url = stripe_js.get("verification_url", "")
                    logger.info(
                        "挑战上下文: intent_id=%s..., site_key=%s..., rqdata_len=%s, verification_url=%s",
                        intent_id[:20],
                        site_key[:20] if site_key else "",
                        len(rqdata or ""),
                        verification_url or "",
                    )
                    self.result.challenge_context = {
                        "intent_id": intent_id,
                        "client_secret": client_secret,
                        "site_key": site_key,
                        "rqdata_len": len(rqdata or ""),
                        "verification_url": verification_url,
                        "stripe_hosted_url": getattr(self, "_stripe_hosted_url", "") or self.checkout_url,
                    }

                    # 可选: 浏览器先手（默认关闭）
                    browser_error = "browser_skipped"
                    if self._enable_browser_challenge and client_secret:
                        browser_result = self._handle_challenge_with_browser(
                            client_secret,
                            site_key=site_key,
                            rqdata=rqdata,
                            verification_url=verification_url,
                            intent_id=intent_id,
                        )
                        if browser_result.get("success"):
                            self.result.success = True
                            self.result.error = ""
                            self.result.confirm_response = browser_result
                            logger.info("浏览器 challenge 处理成功")
                            return self.result
                        browser_error = browser_result.get("error", "browser_failed")
                        logger.warning("浏览器 challenge 处理失败: %s", browser_error)

                    if not self._enable_hcaptcha_solver:
                        # 即使打码关闭，也尽量回捞 confirm 之后的真实 intent 状态与错误码。
                        diag = self._retrieve_intent_diagnostic(
                            intent_id=intent_id,
                            client_secret=client_secret,
                        )
                        if diag:
                            self.result.challenge_context["intent_diagnostic"] = diag
                            status_s = diag.get("status", "")
                            code_s = diag.get("error_code", "")
                            if status_s or code_s:
                                self.result.error = (
                                    "hcaptcha_detected_solver_disabled"
                                    f":status={status_s or 'unknown'}"
                                    f":code={code_s or 'none'}"
                                )
                            else:
                                self.result.error = "hcaptcha_detected_solver_disabled"
                        else:
                            self.result.error = "hcaptcha_detected_solver_disabled"
                        logger.info(
                            "exp_tag=%s",
                            json.dumps(
                                {
                                    **self.result.experiment_tag,
                                    "stage": "challenge",
                                    "outcome": "failed",
                                    "reason": self.result.error,
                                },
                                ensure_ascii=False,
                            ),
                        )
                        return self.result

                    # 打码多轮（Stripe 可能连续下发多轮 challenge）
                    challenge_error = "challenge_params_incomplete"
                    rounds = max(1, int(self._hcaptcha_max_rounds))
                    current_site_key = site_key
                    current_rqdata = rqdata
                    current_verification_url = verification_url
                    current_client_secret = client_secret
                    for idx in range(1, rounds + 1):
                        logger.info("hCaptcha 打码第 %s/%s 轮", idx, rounds)
                        challenge_result = self._handle_stripe_challenge(
                            intent_id=intent_id,
                            client_secret=current_client_secret,
                            site_key=current_site_key,
                            rqdata=current_rqdata,
                            verification_url=current_verification_url,
                        )
                        if challenge_result is True:
                            self.result.success = True
                            self.result.error = ""
                            logger.info("hCaptcha 挑战验证完成，支付成功")
                            return self.result
                        if isinstance(challenge_result, dict):
                            current_site_key = challenge_result.get("site_key", current_site_key)
                            current_rqdata = challenge_result.get("rqdata", current_rqdata)
                            current_verification_url = challenge_result.get("verification_url", current_verification_url)
                            current_client_secret = challenge_result.get("client_secret", current_client_secret)
                            challenge_error = "challenge_next_round_required"
                            logger.info("Stripe 返回新一轮 challenge，上下文已更新")
                            continue
                        challenge_error = str(challenge_result or "challenge_failed")
                        break

                    self.result.error = f"hcaptcha_failed:browser={browser_error};solver={challenge_error}"
                    logger.info(
                        "exp_tag=%s",
                        json.dumps(
                            {
                                **self.result.experiment_tag,
                                "stage": "challenge",
                                "outcome": "failed",
                                "reason": self.result.error,
                            },
                            ensure_ascii=False,
                        ),
                    )
                elif next_action.get("type") == "redirect_to_url":
                    logger.warning("支付需要 3DS 网页验证，无法自动完成")
                    self.result.error = "requires_3ds_redirect"
                else:
                    unknown_action = challenge_type or next_action.get("type")
                    # 对于非 hCaptcha 的 requires_action（例如 stripe_3ds2_fingerprint），
                    # 先尝试浏览器 handleNextAction 自动完成，再决定失败。
                    if self._enable_browser_challenge and intent_client_secret:
                        logger.info("检测到 requires_action=%s，尝试浏览器 handleNextAction", unknown_action)
                        browser_result = self._handle_challenge_with_browser(
                            intent_client_secret,
                            intent_id=(pi.get("id", "") or si.get("id", "")),
                        )
                        if browser_result.get("success"):
                            self.result.success = True
                            self.result.error = ""
                            self.result.confirm_response = browser_result
                            logger.info("浏览器 handleNextAction 成功完成 requires_action")
                            return self.result
                        logger.warning("浏览器 handleNextAction 未完成 requires_action: %s", browser_result.get("error", "unknown"))
                    else:
                        logger.warning(f"未知的 next_action 类型: {unknown_action}")
                    self.result.error = f"requires_action: {unknown_action}"
            elif status in ("succeeded", "complete"):
                self.result.success = True
                logger.info("支付确认成功!")
            else:
                self.result.error = f"支付状态异常: session={status}, pi={pi_status}"
                logger.error(self.result.error)
        else:
            error_msg = ""
            try:
                err_data = resp.json()
                error_msg = err_data.get("error", {}).get("message", resp.text[:300])
            except Exception:
                error_msg = resp.text[:300]
            self.result.error = f"支付确认失败 ({resp.status_code}): {error_msg}"
            logger.error(self.result.error)

        return self.result

    def _poll_payment_page(self, checkout_session_id: str, max_wait: int = 30) -> bool:
        """轮询 payment_pages/poll 等待 session 完成 (amount=0 场景)"""
        import time
        stripe_session = self._stripe_session
        poll_url = f"https://api.stripe.com/v1/payment_pages/{checkout_session_id}/poll"
        for i in range(max_wait // 2):
            time.sleep(2)
            resp = stripe_session.get(
                poll_url,
                params={"key": self.stripe_pk},
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                logger.debug(f"poll #{i+1}: status={status}")
                if status == "complete":
                    return True
            else:
                logger.debug(f"poll #{i+1}: {resp.status_code}")
        return False

    def _submit_challenge_token(
        self,
        intent_id: str,
        client_secret: str,
        verification_url: str,
        captcha_token: str,
        captcha_ekey: str = "",
        solver_user_agent: str = "",
    ):
        """
        提交 challenge token 到 Stripe verify_challenge。
        返回:
          - True
          - dict (下一轮 challenge 上下文)
          - str (失败原因)
        """
        if not (verification_url and captcha_token):
            return "verify_params_incomplete"

        verify_url = f"https://api.stripe.com{verification_url}" if verification_url.startswith("/") else verification_url
        verify_ua = USER_AGENT
        use_solver_ua = (os.getenv("HCAPTCHA_VERIFY_USE_SOLVER_UA", "1") or "1").strip().lower() not in ("0", "false", "no")
        if use_solver_ua and isinstance(solver_user_agent, str) and solver_user_agent.strip():
            verify_ua = solver_user_agent.strip()

        base_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Origin": "https://js.stripe.com",
            "Referer": "https://js.stripe.com/",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": verify_ua,
        }

        # 尽量补齐浏览器风格 Client Hints，减少 token 与请求指纹偏差。
        # 可通过 HCAPTCHA_VERIFY_SEND_CH=0 关闭。
        send_ch = (os.getenv("HCAPTCHA_VERIFY_SEND_CH", "1") or "1").strip().lower() not in ("0", "false", "no")
        if send_ch:
            m_ver = re.search(r"Chrome/(\d+)", verify_ua)
            if m_ver:
                v = m_ver.group(1)
                base_headers["sec-ch-ua"] = f'"Not:A-Brand";v="99", "Google Chrome";v="{v}", "Chromium";v="{v}"'
                base_headers["sec-ch-ua-mobile"] = "?0"
                if "Mac OS X" in verify_ua:
                    base_headers["sec-ch-ua-platform"] = '"macOS"'
                elif "Windows" in verify_ua:
                    base_headers["sec-ch-ua-platform"] = '"Windows"'
                elif "Linux" in verify_ua:
                    base_headers["sec-ch-ua-platform"] = '"Linux"'
        form_data = {
            "challenge_response_token": captcha_token,
            "captcha_vendor_name": "hcaptcha",
            "key": self.stripe_pk,
        }
        # 部分打码平台返回的 ekey 与 Stripe challenge 不稳定，允许通过环境变量控制是否提交。
        # HCAPTCHA_SUBMIT_EKEY:
        #   auto(默认): 有值则提交
        #   with/true/1: 强制提交
        #   without/false/0: 不提交
        ekey_mode = (os.getenv("HCAPTCHA_SUBMIT_EKEY", "auto") or "auto").strip().lower()
        should_send_ekey = (
            ekey_mode in ("with", "true", "1")
            or (ekey_mode == "auto" and bool(captcha_ekey))
        )
        if should_send_ekey:
            form_data["challenge_response_ekey"] = captcha_ekey
        if client_secret:
            form_data["client_secret"] = client_secret

        # 可配置：verify_challenge 是否带 Authorization 头
        # - without(默认): 与浏览器抓包一致，不带 Authorization，仅 body 里传 key
        # - with: 带 Authorization: Bearer <pk_live...>
        # - auto: 先 without；若 401/403 再 with 重试一次（同 token）
        auth_mode = (os.getenv("HCAPTCHA_VERIFY_AUTH_HEADER", "without") or "without").strip().lower()
        verify_headers_list = []
        if auth_mode == "with":
            h = dict(base_headers)
            h["Authorization"] = f"Bearer {self.stripe_pk}"
            verify_headers_list = [("with_auth", h)]
        elif auth_mode == "auto":
            h_no = dict(base_headers)
            h_yes = dict(base_headers)
            h_yes["Authorization"] = f"Bearer {self.stripe_pk}"
            verify_headers_list = [("no_auth", h_no), ("with_auth", h_yes)]
        else:
            verify_headers_list = [("no_auth", dict(base_headers))]

        resp = None
        data = None
        last_parse_err = ""
        for hdr_label, headers in verify_headers_list:
            logger.info(
                "[支付 5/5] 提交 hCaptcha 挑战验证: %s... (auth_header=%s, ua=%s)",
                (intent_id or "")[:20],
                hdr_label,
                (headers.get("User-Agent", "") or "")[:64],
            )

            # 调试落盘：真实 verify 请求体（避免无报错盲区）
            try:
                os.makedirs("test_outputs", exist_ok=True)
                ts = int(time.time() * 1000)
                req_path = f"test_outputs/solver_verify_req_{ts}.txt"
                with open(req_path, "w", encoding="utf-8") as f:
                    f.write(urlencode(form_data, doseq=False))
                logger.info("已保存 solver verify 请求体: %s", req_path)
            except Exception:
                pass

            resp = self._stripe_session.post(verify_url, headers=headers, data=form_data, timeout=60)
            logger.info("verify_challenge 状态: %s", resp.status_code)

            try:
                data = resp.json()
            except Exception:
                data = None
                last_parse_err = resp.text[:120]

            try:
                os.makedirs("test_outputs", exist_ok=True)
                ts = int(time.time() * 1000)
                resp_path = f"test_outputs/solver_verify_resp_{ts}.json"
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                logger.info("已保存 solver verify 响应: %s", resp_path)
            except Exception:
                pass

            # auto 模式下仅对鉴权类错误尝试下一套头；其余情况不重复提交同 token
            if auth_mode == "auto" and resp.status_code in (401, 403):
                continue
            break

        if data is None:
            return f"verify_response_parse_failed:{last_parse_err}"

        # 保留 verify 响应与最终响应，便于排障
        self.result.verify_response = data
        self.result.confirm_response = data

        if resp.status_code != 200:
            err = (data.get("error") or {})
            err_code = err.get("code", "")
            err_msg = err.get("message", "") or str(data)[:160]
            if err_code:
                return f"verify_http_{resp.status_code}:{err_code}"
            return f"verify_http_{resp.status_code}:{err_msg[:80]}"

        status = data.get("status", "")
        if resp.status_code == 200:
            last_err = data.get("last_setup_error") or data.get("last_payment_error") or {}
            if last_err:
                logger.info(
                    "verify_challenge 结果: intent_status=%s error_type=%s error_code=%s message=%s",
                    status or "",
                    last_err.get("type", ""),
                    last_err.get("code", ""),
                    (last_err.get("message", "") or "")[:160],
                )
            else:
                logger.info("verify_challenge 结果: intent_status=%s", status or "")
        if status in ("succeeded", "processing"):
            return True

        if status == "requires_action":
            next_action = data.get("next_action", {}) or {}
            sdk_info = next_action.get("use_stripe_sdk", {}) or {}
            if sdk_info.get("type") == "intent_confirmation_challenge":
                stripe_js = sdk_info.get("stripe_js", {}) or {}
                return {
                    "site_key": stripe_js.get("site_key", ""),
                    "rqdata": stripe_js.get("rqdata", ""),
                    "verification_url": stripe_js.get("verification_url", ""),
                    "client_secret": data.get("client_secret", client_secret),
                }
            return f"verify_requires_action:{next_action.get('type') or sdk_info.get('type') or 'unknown'}"

        if status == "requires_payment_method":
            last_err = data.get("last_setup_error") or data.get("last_payment_error") or {}
            code = last_err.get("code") or "requires_payment_method"
            return f"verify_auth_failed:{code}"

        return f"verify_unexpected_status:{status or 'unknown'}"

    def _retrieve_intent_diagnostic(self, intent_id: str, client_secret: str) -> dict:
        """
        在 challenge 未解成功时，主动拉取 Intent 最新状态，补齐 confirm 之后的错误信息。

        返回示例:
        {
          "intent_id": "...",
          "status": "requires_action",
          "error_type": "...",
          "error_code": "...",
          "error_message": "...",
          "next_action_type": "...",
          "sdk_type": "...",
          "http_status": 200
        }
        """
        if not intent_id:
            return {}

        is_setup_intent = str(intent_id).startswith("seti_")
        endpoint = "setup_intents" if is_setup_intent else "payment_intents"
        url = f"https://api.stripe.com/v1/{endpoint}/{intent_id}"

        params = {"key": self.stripe_pk}
        if client_secret:
            params["client_secret"] = client_secret

        headers = {
            "Authorization": f"Bearer {self.stripe_pk}",
            "Accept": "application/json",
            "Origin": "https://js.stripe.com",
            "Referer": "https://js.stripe.com/",
            "User-Agent": USER_AGENT,
        }

        try:
            resp = self._stripe_session.get(url, headers=headers, params=params, timeout=30)
        except Exception as e:
            logger.warning("intent 诊断请求异常: %s", e)
            return {"intent_id": intent_id, "diag_error": f"request_exception:{e}"}

        try:
            data = resp.json()
        except Exception:
            data = {}

        diag = {
            "intent_id": intent_id,
            "http_status": resp.status_code,
        }

        if resp.status_code != 200:
            err = (data.get("error") or {}) if isinstance(data, dict) else {}
            diag.update(
                {
                    "diag_error": f"http_{resp.status_code}",
                    "error_type": err.get("type", ""),
                    "error_code": err.get("code", ""),
                    "error_message": (err.get("message", "") or str(data)[:200]),
                }
            )
            logger.info(
                "intent 诊断失败: intent=%s http=%s code=%s msg=%s",
                intent_id[:20],
                resp.status_code,
                diag.get("error_code", ""),
                (diag.get("error_message", "") or "")[:160],
            )
            return diag

        status = data.get("status", "")
        next_action = data.get("next_action", {}) or {}
        sdk_info = next_action.get("use_stripe_sdk", {}) if isinstance(next_action, dict) else {}
        last_err = data.get("last_setup_error") or data.get("last_payment_error") or {}

        diag.update(
            {
                "status": status,
                "next_action_type": next_action.get("type", "") if isinstance(next_action, dict) else "",
                "sdk_type": sdk_info.get("type", "") if isinstance(sdk_info, dict) else "",
                "error_type": last_err.get("type", ""),
                "error_code": last_err.get("code", ""),
                "error_message": (last_err.get("message", "") or "")[:300],
            }
        )
        logger.info(
            "intent 诊断: intent=%s status=%s next_action=%s sdk=%s err_code=%s",
            intent_id[:20],
            status or "",
            diag.get("next_action_type", ""),
            diag.get("sdk_type", ""),
            diag.get("error_code", ""),
        )
        return diag

    # ── Step 5a: 浏览器处理 hCaptcha（可选） ──
    def _handle_challenge_with_browser(
        self,
        client_secret: str,
        site_key: str = "",
        rqdata: str = "",
        verification_url: str = "",
        intent_id: str = "",
    ) -> dict:
        """可选地使用浏览器执行 handleNextAction 或 direct hCaptcha。失败返回 {success: False, error: ...}。"""
        if not client_secret:
            return {"success": False, "error": "missing_client_secret"}

        try:
            from browser_challenge import BrowserChallengeSolver
        except Exception:
            return {"success": False, "error": "browser_challenge_not_available"}

        # Stripe/hCaptcha 在无头模式更易被风控，默认使用 headed + Xvfb
        headless = os.getenv("BROWSER_CHALLENGE_HEADLESS", "0") not in ("0", "false", "False")
        timeout = int(os.getenv("BROWSER_CHALLENGE_TIMEOUT", "120"))
        solver = BrowserChallengeSolver(
            stripe_pk=self.stripe_pk,
            proxy=self._stripe_proxy,
            headless=headless,
        )

        # 先走 Stripe 官方 handleNextAction；失败时可选走 direct hCaptcha + verify_challenge
        try:
            result = solver.solve(
                client_secret,
                timeout=timeout,
                challenge_url=(getattr(self, "_stripe_hosted_url", "") or self.checkout_url or ""),
            ) or {}
            if result.get("success"):
                return result

            # 若 handleNextAction 未完成，但浏览器内已捕获 hCaptcha token，优先直接提交 verify_challenge。
            captured_token = result.get("hcaptcha_token", "")
            captured_ekey = result.get("hcaptcha_ekey", "")
            browser_err = result.get("error", "browser_handle_next_action_failed")
            browser_code = (result.get("error_code", "") or "").strip()
            # 若浏览器内已执行 verify_challenge 且明确返回认证失败，
            # 不再重复提交同一 token，避免污染日志为 “no valid challenge associated...”
            if browser_code in ("setup_intent_authentication_failure", "payment_intent_authentication_failure"):
                return {
                    "success": False,
                    "error": f"browser_verify_failed:{browser_code}",
                    "raw_browser_result": result,
                }
            if captured_token and verification_url:
                logger.info(
                    "[Browser] 使用浏览器捕获 token 提交 verify_challenge (token_len=%s, ekey=%s, source=%s)",
                    len(captured_token),
                    (captured_ekey or "")[:24],
                    (result.get("hcaptcha_token_source") or "")[:48],
                )
                challenge_result = self._submit_challenge_token(
                    intent_id=intent_id,
                    client_secret=client_secret,
                    verification_url=verification_url,
                    captcha_token=captured_token,
                    captcha_ekey=captured_ekey,
                )
                if challenge_result is True:
                    return {"success": True, "mode": "browser_captured_token_verify"}
                if isinstance(challenge_result, dict):
                    return {"success": False, "error": "browser_next_round_required", "next_challenge": challenge_result}
                return {"success": False, "error": f"browser_captured_token_verify_failed:{challenge_result}"}

            if not (site_key and verification_url):
                return {"success": False, "error": browser_err}

            # 可选关闭 direct fallback（默认开启）。
            # 手动交互调试/CTF 场景下，关闭可避免重复等待 invisible challenge 超时。
            enable_direct_fallback = os.getenv("BROWSER_DIRECT_FALLBACK", "1") not in ("0", "false", "False")
            if not enable_direct_fallback:
                return {"success": False, "error": browser_err}

            direct = solver.solve_hcaptcha_direct(
                site_key=site_key,
                site_url=(getattr(self, "_stripe_hosted_url", "") or self.checkout_url or "https://js.stripe.com"),
                rqdata=rqdata,
                timeout=timeout,
            ) or {}
            if not direct.get("success"):
                return {"success": False, "error": f"{browser_err};direct={direct.get('error', 'direct_failed')}"}

            challenge_result = self._submit_challenge_token(
                intent_id=intent_id,
                client_secret=client_secret,
                verification_url=verification_url,
                captcha_token=direct.get("token", ""),
                captcha_ekey=direct.get("ekey", ""),
            )
            if challenge_result is True:
                return {"success": True, "mode": "browser_direct_verify"}
            if isinstance(challenge_result, dict):
                # 浏览器拿到 token 但 Stripe 继续下发下一轮 challenge
                return {"success": False, "error": "browser_next_round_required", "next_challenge": challenge_result}
            return {"success": False, "error": f"browser_verify_failed:{challenge_result}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Step 5b: YesCaptcha 处理 Stripe challenge ──
    def _handle_stripe_challenge(
        self,
        intent_id: str,
        client_secret: str,
        site_key: str,
        rqdata: str,
        verification_url: str,
    ):
        """
        解决 intent_confirmation_challenge:
          1) 打码拿 token/ekey
          2) 调 verify_challenge
        返回:
          - True: 验证后成功
          - dict: Stripe 要求下一轮 challenge，上下文已更新
          - str/False: 失败原因
        """
        if not (intent_id and site_key and verification_url):
            return "challenge_params_incomplete"
        if not self.config.captcha.client_key:
            return "captcha_key_missing"

        solver = CaptchaSolver(
            api_url=self.config.captcha.api_url,
            client_key=self.config.captcha.client_key,
        )

        site_url = getattr(self, "_stripe_hosted_url", "") or self.checkout_url or "https://js.stripe.com"
        proxy_mode = (os.getenv("CAPTCHA_PROXY_MODE", "auto") or "auto").strip().lower()
        proxies_to_try = []

        def _is_local_proxy(proxy_url: str) -> bool:
            """
            判断代理是否为本机/内网地址。
            这类代理对打码平台不可达（平台侧无法连接 127.0.0.1 / 内网），
            不应使用 HCaptchaTask 代理模式。
            """
            if not proxy_url:
                return False
            try:
                parsed = urlsplit(proxy_url)
                host = (parsed.hostname or "").strip().lower()
                if not host:
                    return False
                if host in ("localhost",):
                    return True
                ip = ipaddress.ip_address(host)
                return bool(ip.is_loopback or ip.is_private or ip.is_link_local)
            except Exception:
                # 不是纯 IP（域名）时，保守视作非本地
                return False

        # 可选：单独给打码平台配置代理（不影响主流程直连）
        # 例如：
        #   CAPTCHA_PROXY_URL=http://<本机公网IP>:8899
        captcha_proxy_override = (os.getenv("CAPTCHA_PROXY_URL", "") or "").strip()
        captcha_proxy = captcha_proxy_override or (self._stripe_proxy or "")
        local_proxy = _is_local_proxy(captcha_proxy)

        if proxy_mode == "proxy":
            if captcha_proxy and not local_proxy:
                proxies_to_try = [captcha_proxy]
            else:
                return "captcha_proxy_mode_requires_public_proxy"
        elif proxy_mode == "proxyless":
            proxies_to_try = [""]
        else:  # auto
            if captcha_proxy and not local_proxy:
                proxies_to_try.append(captcha_proxy)
            elif captcha_proxy and local_proxy:
                logger.info("检测到本地/内网代理，打码服务跳过 proxy 模式，优先 proxyless")
            proxies_to_try.append("")

        # 可配置：同一模式下重试次数（默认 2）
        try:
            solver_retries = max(1, int(os.getenv("CAPTCHA_SOLVER_RETRIES", "2")))
        except Exception:
            solver_retries = 2

        # 可配置：hCaptcha invisible 策略
        # - visible/false/0: 仅按可见题提交
        # - invisible/true/1: 仅按 invisible 提交
        # - auto(默认): 先 visible 再 invisible，提升兼容性
        invisible_mode = (os.getenv("HCAPTCHA_INVISIBLE_MODE", "auto") or "auto").strip().lower()
        if invisible_mode in ("1", "true", "invisible"):
            invisible_candidates = [True]
        elif invisible_mode in ("0", "false", "visible"):
            invisible_candidates = [False]
        else:
            invisible_candidates = [False, True]

        # 可配置：Enterprise 模式
        # - true/1/enterprise(默认): 按 Enterprise 解题
        # - false/0: 按非 Enterprise 解题
        # - auto/both: 两种都尝试
        enterprise_mode = (os.getenv("HCAPTCHA_ENTERPRISE_MODE", "enterprise") or "enterprise").strip().lower()
        if enterprise_mode in ("0", "false", "non-enterprise", "normal"):
            enterprise_candidates = [False]
        elif enterprise_mode in ("auto", "both"):
            enterprise_candidates = [True, False]
        else:
            enterprise_candidates = [True]

        # 可配置：网站 URL 取值策略（部分平台对 URL 非常敏感）
        # - full: 使用完整 hosted_url
        # - no_fragment: 去掉 #fragment
        # - origin: 仅使用 scheme://host/
        # - auto(默认): Stripe CDN + full + no_fragment + origin
        split_url = urlsplit(site_url)
        site_url_no_fragment = urlunsplit((split_url.scheme, split_url.netloc, split_url.path, split_url.query, ""))
        site_url_origin = f"{split_url.scheme}://{split_url.netloc}/" if split_url.scheme and split_url.netloc else site_url
        site_url_mode = (os.getenv("HCAPTCHA_SITEURL_MODE", "auto") or "auto").strip().lower()
        if site_url_mode == "no_fragment":
            site_url_candidates = [site_url_no_fragment]
        elif site_url_mode == "origin":
            site_url_candidates = [site_url_origin]
        elif site_url_mode == "auto":
            site_url_candidates = [site_url, site_url_no_fragment, site_url_origin]
        else:
            site_url_candidates = [site_url]

        # Stripe 的 hCaptcha 实际承载域常为 b.stripecdn.com；若仅用 checkout URL，打码可解但 token 可能无效。
        # 默认将其置顶尝试，可通过 HCAPTCHA_SITEURL_INCLUDE_STRIPECDN=0 关闭。
        include_stripe_cdn = (os.getenv("HCAPTCHA_SITEURL_INCLUDE_STRIPECDN", "1") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        if include_stripe_cdn:
            stripe_cdn_url = "https://b.stripecdn.com/"
            site_url_candidates = [stripe_cdn_url] + site_url_candidates

        # 允许追加自定义站点 URL（逗号分隔）
        # 例如：HCAPTCHA_SITEURL_EXTRA=https://newassets.hcaptcha.com/,https://js.stripe.com/
        extra_raw = (os.getenv("HCAPTCHA_SITEURL_EXTRA", "") or "").strip()
        if extra_raw:
            extras = [u.strip() for u in extra_raw.split(",") if u.strip()]
            if extras:
                site_url_candidates.extend(extras)
        # 去重并保持顺序
        _seen = set()
        site_url_candidates = [u for u in site_url_candidates if u and (u not in _seen and not _seen.add(u))]

        # 可配置：是否要求打码结果必须带 ekey
        # - auto(默认): Stripe hCaptcha(site_key=c7faac4c-...)时要求
        # - true/1: 总是要求
        # - false/0: 不要求
        require_ekey_mode = (os.getenv("HCAPTCHA_REQUIRE_EKEY", "auto") or "auto").strip().lower()
        if require_ekey_mode in ("1", "true", "yes", "with"):
            require_ekey = True
        elif require_ekey_mode in ("0", "false", "no", "without"):
            require_ekey = False
        else:
            require_ekey = str(site_key or "").startswith("c7faac4c-")

        last_error = ""
        for px in proxies_to_try:
            mode = "proxy" if px else "proxyless"
            for solve_site_url in site_url_candidates:
                for is_enterprise in enterprise_candidates:
                    for is_invisible in invisible_candidates:
                        for retry_idx in range(1, solver_retries + 1):
                            logger.info(
                                "hCaptcha 打码模式: %s, enterprise=%s, invisible=%s, attempt=%s/%s, site_url=%s",
                                mode,
                                is_enterprise,
                                is_invisible,
                                retry_idx,
                                solver_retries,
                                solve_site_url[:120],
                            )
                            captcha_result = solver.solve_hcaptcha(
                                site_key=site_key,
                                site_url=solve_site_url,
                                rqdata=rqdata,
                                user_agent=USER_AGENT,
                                proxy=px,
                                timeout=int(os.getenv("HCAPTCHA_TIMEOUT", "120")),
                                is_invisible=is_invisible,
                                is_enterprise=is_enterprise,
                            )
                            if captcha_result:
                                captcha_token = captcha_result.get("token", "")
                                captcha_ekey = captcha_result.get("ekey", "")
                                solver_ua = (captcha_result.get("user_agent", "") or "").strip()
                                ekey_source = (captcha_result.get("ekey_source", "") or "").strip()
                                if not captcha_token:
                                    last_error = "captcha_token_missing"
                                    continue
                                # 诊断防呆：Stripe challenge token 通常为 P1_/E1_。
                                # 若供应商返回格式异常，直接丢弃，避免消耗一次 verify_challenge。
                                if not str(captcha_token).startswith("P1_"):
                                    last_error = "captcha_token_format_invalid"
                                    logger.warning(
                                        "打码 token 格式异常，跳过本次提交: mode=%s site_url=%s token_prefix=%s",
                                        mode,
                                        solve_site_url[:120],
                                        str(captcha_token)[:16],
                                    )
                                    continue
                                if require_ekey and not captcha_ekey:
                                    last_error = "captcha_ekey_missing"
                                    logger.warning(
                                        "打码返回缺少 ekey，跳过本次 token: mode=%s site_url=%s enterprise=%s invisible=%s",
                                        mode,
                                        solve_site_url[:120],
                                        is_enterprise,
                                        is_invisible,
                                    )
                                    continue
                                if captcha_ekey and not str(captcha_ekey).startswith("E1_"):
                                    last_error = "captcha_ekey_format_invalid"
                                    logger.warning(
                                        "打码 ekey 格式异常，跳过本次 token: source=%s prefix=%s mode=%s site_url=%s",
                                        ekey_source or "unknown",
                                        str(captcha_ekey)[:16],
                                        mode,
                                        solve_site_url[:120],
                                    )
                                    continue

                                submit_result = self._submit_challenge_token(
                                    intent_id=intent_id,
                                    client_secret=client_secret,
                                    verification_url=verification_url,
                                    captcha_token=captcha_token,
                                    captcha_ekey=captcha_ekey,
                                    solver_user_agent=solver_ua,
                                )

                                # 成功 / 下一轮 challenge 直接返回
                                if submit_result is True or isinstance(submit_result, dict):
                                    return submit_result

                                last_error = str(submit_result or "challenge_failed")

                                # 认证失败通常意味着当前 payment attempt 已失效：
                                # 再次提交同一轮 challenge 往往得到
                                # "There is no valid challenge associated with the current payment attempt."
                                # 直接返回给上层，让 confirm 全链路重建 PM 并获取新 challenge。
                                if "verify_auth_failed" in last_error:
                                    logger.warning(
                                        "verify_auth_failed，结束本轮 challenge，交由上层 confirm 重建: mode=%s invisible=%s (%s/%s)",
                                        mode,
                                        is_invisible,
                                        retry_idx,
                                        solver_retries,
                                    )
                                    return last_error

                                if "verify_http_400" in last_error and "valid challenge" in last_error.lower():
                                    logger.warning(
                                        "challenge 已失效/不匹配，结束本轮并交由上层 confirm 重建: %s",
                                        last_error,
                                    )
                                    return last_error

                                # 其它失败直接返回
                                return last_error

                            # 本轮打码失败，继续同模式后续重试
                            if solver.last_error:
                                last_error = f"captcha_provider_error:{solver.last_error}"
                            else:
                                last_error = "captcha_unsolved_or_provider_error"

        if not last_error:
            last_error = "captcha_unsolved_or_provider_error"
        return last_error

    # ── Step 4b: 代理切换重试 confirm ──
    def confirm_payment_with_proxy(self, checkout_session_id: str, proxy: str = None) -> PaymentResult:
        """
        使用指定代理重新走 tokenize + init + confirm (代理开关策略)。
        PaymentMethod 在首次 confirm 后已被消费，必须重新创建。
        """
        logger.info(f"[支付 4b] 代理切换全流程重试 (proxy={proxy})...")

        # 1) 重新创建 PaymentMethod (用新代理)
        stripe_session = create_http_session(proxy=proxy)
        card = self.config.card
        billing = self.config.billing
        fp = self.fingerprint.get_params()

        pm_form = {
            "type": "card",
            "card[number]": card.number,
            "card[cvc]": card.cvc,
            "card[exp_month]": card.exp_month,
            "card[exp_year]": card.exp_year,
            "billing_details[name]": billing.name,
            "billing_details[email]": billing.email or self.auth.email,
            "billing_details[address][country]": billing.country,
            "billing_details[address][line1]": billing.address_line1,
            "billing_details[address][line2]": getattr(billing, "address_line2", "") or "",
            "billing_details[address][city]": getattr(billing, "address_city", "") or "",
            "billing_details[address][state]": billing.address_state,
            "billing_details[address][postal_code]": billing.postal_code,
            "allow_redisplay": "always",
            "guid": fp["guid"],
            "muid": fp["muid"],
            "sid": fp["sid"],
            "payment_user_agent": f"stripe.js/{self.config.stripe_build_hash}; stripe-js-v3/{self.config.stripe_build_hash}; checkout",
        }

        headers = {
            "Authorization": f"Bearer {self.stripe_pk}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Origin": "https://js.stripe.com",
            "Referer": "https://js.stripe.com/",
            "User-Agent": USER_AGENT,
        }

        resp = stripe_session.post(
            "https://api.stripe.com/v1/payment_methods",
            headers=headers,
            data=pm_form,
            timeout=30,
        )
        if resp.status_code != 200:
            self.result.error = f"代理切换: 创建 PaymentMethod 失败 ({resp.status_code})"
            logger.error(self.result.error)
            return self.result

        new_pm_id = resp.json().get("id", "")
        logger.info(f"代理切换: 新 PaymentMethod: {new_pm_id[:20]}...")

        # 2) 重新 init (获取新 eid/checksum, 但复用原始 expected_amount)
        init_url = f"https://api.stripe.com/v1/payment_pages/{checkout_session_id}/init"
        init_form = {"key": self.stripe_pk, "browser_locale": "en"}
        init_resp = stripe_session.post(init_url, headers=headers, data=init_form, timeout=30)
        eid = ""
        checksum = ""
        # 复用首次 confirm 的 expected_amount (代理 IP 可能导致税率不同)
        expected = getattr(self, '_expected_amount', "0")
        if init_resp.status_code == 200:
            init_data = init_resp.json()
            eid = init_data.get("eid", "")
            checksum = init_data.get("init_checksum", "")
            logger.info(f"代理切换: init eid={eid[:10]}... expected_amount={expected} (复用原值)")

        # 3) Confirm
        form_data = {
            "payment_method": new_pm_id,
            "guid": fp["guid"],
            "muid": fp["muid"],
            "sid": fp["sid"],
            "expected_amount": expected,
            "key": self.stripe_pk,
        }
        if eid:
            form_data["eid"] = eid
        if checksum:
            form_data["init_checksum"] = checksum

        url = f"https://api.stripe.com/v1/payment_pages/{checkout_session_id}/confirm"
        resp = stripe_session.post(url, headers=headers, data=form_data, timeout=60)

        self.result.confirm_status = str(resp.status_code)
        try:
            self.result.confirm_response = resp.json()
        except Exception:
            self.result.confirm_response = {"raw": resp.text[:500]}

        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "")
            pi = data.get("payment_intent") or {}
            pi_status = pi.get("status", "")
            next_action = pi.get("next_action", {})

            if status == "complete" or (status == "open" and pi_status == "succeeded"):
                self.result.success = True
                self.result.error = ""
                logger.info("代理切换 confirm 支付成功!")
            elif pi_status == "requires_action" and next_action:
                sdk_info = next_action.get("use_stripe_sdk", {})
                challenge_type = sdk_info.get("type", "")
                if challenge_type == "intent_confirmation_challenge":
                    logger.info("代理切换 confirm 触发 hCaptcha，按 CTF 规则直接判负")
                    self.result.error = "hcaptcha_detected_auto_fail"
                else:
                    self.result.error = f"代理切换 confirm requires_action: {challenge_type}"
            else:
                self.result.error = f"代理切换 confirm 状态异常: session={status}, pi={pi_status}"
        else:
            error_msg = ""
            try:
                err_data = resp.json()
                error_msg = err_data.get("error", {}).get("message", resp.text[:300])
            except Exception:
                error_msg = resp.text[:300]
            self.result.error = f"代理切换 confirm 失败 ({resp.status_code}): {error_msg}"
            logger.error(self.result.error)

        return self.result

    # ── 完整支付流程 ──
    def run_payment(self) -> PaymentResult:
        """执行支付链路。OPENAI_LINK_ONLY=1 时仅生成 OpenAI 订阅链接，不执行 Stripe confirm。"""
        try:
            run_promo_id = self._exp_promo_id or self.config.team_plan.promo_campaign_id
            if run_promo_id in ("none", "null", "off", "-"):
                run_promo_id = ""
            logger.info(
                "exp_tag=%s",
                json.dumps(
                    {
                        "mode": "ctf",
                        **self.result.experiment_tag,
                        "run_id": self._exp_run_id,
                        "promo_variant": self._exp_promo_variant,
                        "promo_id": run_promo_id,
                        "stage": "run_start",
                        "strict_confirm": self._strict_confirm,
                    },
                    ensure_ascii=False,
                ),
            )
            cs_id = self.create_checkout_session()
            if self._openai_link_only and not self._strict_confirm:
                self.result.success = False
                self.result.checkout_session_id = cs_id
                self.result.confirm_status = "openai_link_only"
                self.result.error = "requires_confirmation_stage"
                logger.info("OPENAI_LINK_ONLY 已启用：仅生成 OpenAI 订阅链接，未执行最终验证确认阶段")
                return self.result
            if self._openai_link_only and self._strict_confirm:
                logger.warning("STRICT_CONFIRM=1: 忽略 OPENAI_LINK_ONLY，继续执行最终确认阶段")

            self.extract_stripe_pk(self.checkout_url)
            self.fetch_payment_page_details(cs_id)
            if self._init_indicates_hcaptcha():
                if self._init_hcaptcha_auto_fail:
                    logger.info("init 预判命中 hCaptcha，INIT_HCAPTCHA_AUTO_FAIL=1，跳过后续并判负")
                    self.result.confirm_status = "skipped_hcaptcha_predicted"
                    self.result.error = "hcaptcha_predicted_from_init_auto_fail"
                    logger.info(
                        "exp_tag=%s",
                        json.dumps(
                            {
                                **self.result.experiment_tag,
                                "stage": "challenge",
                                "outcome": "failed",
                                "reason": "hcaptcha_predicted_from_init_auto_fail",
                            },
                            ensure_ascii=False,
                        ),
                    )
                    logger.info(
                        "exp_tag=%s",
                        json.dumps(
                            {
                                **self.result.experiment_tag,
                                "stage": "run_end",
                                "outcome": "failed",
                                "reason": self.result.error,
                            },
                            ensure_ascii=False,
                        ),
                    )
                    return self.result
                logger.warning("init 预判命中 hCaptcha，但 INIT_HCAPTCHA_AUTO_FAIL=0，继续后续链路")

            # confirm 可重试：当 challenge 验证失败（尤其 verify_auth_failed）时，重建 PM 再试
            try:
                max_confirm_attempts = max(1, int(os.getenv("CONFIRM_MAX_ATTEMPTS", "2")))
            except Exception:
                max_confirm_attempts = 2

            result = self.result
            for attempt in range(1, max_confirm_attempts + 1):
                if attempt > 1:
                    logger.warning("[支付] 第 %s/%s 次 confirm 重试", attempt, max_confirm_attempts)
                    # 重试前刷新 init 上下文（eid/checksum）和期望金额
                    self.fetch_payment_page_details(cs_id)
                self.fetch_stripe_fingerprint()
                self.payment_method_id = self.create_payment_method()
                result = self.confirm_payment(cs_id)
                if result.success:
                    break

                retryable = any(
                    k in (result.error or "")
                    for k in (
                        "verify_auth_failed",
                        "hcaptcha_failed",
                        "captcha_unsolved_or_provider_error",
                        "hcaptcha_detected_solver_disabled",
                    )
                )
                if not retryable:
                    break
                if attempt >= max_confirm_attempts:
                    break

            logger.info(
                "exp_tag=%s",
                json.dumps(
                    {
                        **self.result.experiment_tag,
                        "stage": "run_end",
                        "outcome": "success" if result.success else "failed",
                        "reason": result.error,
                    },
                    ensure_ascii=False,
                ),
            )
            return result
        except Exception as e:
            self.result.error = str(e)
            logger.error(f"支付流程异常: {e}")
            logger.info(
                "exp_tag=%s",
                json.dumps(
                    {
                        **self.result.experiment_tag,
                        "stage": "run_exception",
                        "outcome": "failed",
                        "reason": str(e)[:200],
                    },
                    ensure_ascii=False,
                ),
            )
            return self.result

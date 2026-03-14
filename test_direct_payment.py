"""
测试 - Stripe 直连支付 (ChatGPT 走代理, Stripe 直连)
"""
import logging
import json
import sys
import glob

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_direct")

from config import Config, CardInfo, BillingInfo, CaptchaConfig
from auth_flow import AuthResult
from payment_flow import PaymentFlow

# 加载最近的凭证
cred_files = sorted(glob.glob("test_outputs/credentials_*.json"))
if not cred_files:
    print("没有找到保存的凭证")
    sys.exit(1)

latest = cred_files[-1]
logger.info(f"使用凭证: {latest}")
cred = json.load(open(latest))

# 构建 AuthResult
auth = AuthResult()
auth.email = cred["email"]
auth.password = cred.get("password", "")
auth.session_token = cred["session_token"]
auth.access_token = cred["access_token"]
auth.device_id = cred.get("device_id", "")

cfg = Config()
cfg.proxy = "http://172.25.16.1:7897"  # ChatGPT 用代理 (Windows 宿主机)
cfg.card = CardInfo(
    number="4462220004624356",
    cvc="173",
    exp_month="03",
    exp_year="2029",
)
cfg.billing = BillingInfo(
    name="Test User",
    email=auth.email,
    country="GB",
    currency="GBP",
    address_line1="Langley House",
    address_state="England",
    postal_code="N2 8EY",
)
cfg.captcha = CaptchaConfig(
    api_url="https://api.yescaptcha.com",
    client_key="27e2aa9da9a236b2a6cfcc3fa0f045fdec2a3633104361",
)

pf = PaymentFlow(cfg, auth)

try:
    result = pf.run_payment()
    logger.info("=" * 60)
    logger.info(f"支付结果:")
    logger.info(f"  状态码: {result.confirm_status}")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  错误: {result.error}")
    if result.confirm_response:
        resp_str = json.dumps(result.confirm_response, indent=2, ensure_ascii=False)
        logger.info(f"  响应: {resp_str[:1500]}")
    if result.checkout_data:
        logger.info(f"  Checkout data keys: {list(result.checkout_data.keys())}")
except Exception as e:
    logger.error(f"异常: {e}")
    import traceback
    traceback.print_exc()

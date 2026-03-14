"""
自动化绑卡支付 - Streamlit UI
运行: streamlit run ui.py --server.address 0.0.0.0 --server.port 8503
"""
import json
import logging
import os
import sys
import traceback
import threading
from collections import deque

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, CardInfo, BillingInfo, CaptchaConfig
from mail_provider import MailProvider
from auth_flow import AuthFlow, AuthResult
from payment_flow import PaymentFlow
from logger import ResultStore

OUTPUT_DIR = "test_outputs"

import re as _re

# 国家名/后缀 → (country_code, currency) 映射
_COUNTRY_ALIAS = {
    "UK": ("GB", "GBP"), "GB": ("GB", "GBP"), "England": ("GB", "GBP"), "United Kingdom": ("GB", "GBP"),
    "US": ("US", "USD"), "USA": ("US", "USD"), "United States": ("US", "USD"),
    "DE": ("DE", "EUR"), "Germany": ("DE", "EUR"),
    "JP": ("JP", "JPY"), "Japan": ("JP", "JPY"),
    "FR": ("FR", "EUR"), "France": ("FR", "EUR"),
    "SG": ("SG", "SGD"), "Singapore": ("SG", "SGD"),
    "HK": ("HK", "HKD"), "Hong Kong": ("HK", "HKD"),
    "KR": ("KR", "KRW"), "Korea": ("KR", "KRW"),
    "AU": ("AU", "AUD"), "Australia": ("AU", "AUD"),
    "CA": ("CA", "CAD"), "Canada": ("CA", "CAD"),
    "NL": ("NL", "EUR"), "Netherlands": ("NL", "EUR"),
    "IT": ("IT", "EUR"), "Italy": ("IT", "EUR"),
    "ES": ("ES", "EUR"), "Spain": ("ES", "EUR"),
    "CH": ("CH", "CHF"), "Switzerland": ("CH", "CHF"),
}


def _parse_card_text(text: str) -> dict:
    """从粘贴文本中解析卡号、有效期、CVV、账单地址"""
    result = {}
    lines = [l.strip() for l in text.strip().splitlines()]

    # 卡号: 连续 13-19 位数字 (可有空格)
    for line in lines:
        digits_only = line.replace(" ", "").replace("-", "")
        if digits_only.isdigit() and 13 <= len(digits_only) <= 19:
            result["card_number"] = digits_only
            break

    # 有效期: MM/YY 或 MM/YYYY
    for line in lines:
        m = _re.search(r'\b(0[1-9]|1[0-2])\s*/\s*(\d{2,4})\b', line)
        if m:
            result["exp_month"] = m.group(1)
            yr = m.group(2)
            if len(yr) == 2:
                yr = "20" + yr
            result["exp_year"] = yr
            break

    # CVV: "CVV" 后面或下一行的 3-4 位数字
    for i, line in enumerate(lines):
        if _re.search(r'(?i)\b(?:cvv|cvc|安全码)\b', line):
            m = _re.search(r'\b(\d{3,4})\b', line)
            if m:
                result["cvv"] = m.group(1)
            elif i + 1 < len(lines):
                m2 = _re.search(r'\b(\d{3,4})\b', lines[i + 1])
                if m2:
                    result["cvv"] = m2.group(1)
            break

    # 账单地址: "账单地址" 或 "billing address" 后面的地址行
    addr_text = ""
    for i, line in enumerate(lines):
        if _re.search(r'(?i)账单地址|billing\s*address', line):
            # 地址可能在同一行冒号后、或后续行
            after = _re.sub(r'(?i)^.*?(账单地址|billing\s*address)\s*[:：]?\s*', '', line).strip()
            if after and len(after) > 3:
                addr_text = after
            else:
                # 跳过空行和"复制"标签，找到实际地址
                for j in range(i + 1, min(i + 5, len(lines))):
                    candidate = lines[j]
                    if candidate and candidate not in ("复制", "copy", ""):
                        addr_text = candidate
                        break
            break

    if addr_text:
        result["raw_address"] = addr_text
        parts = [p.strip() for p in addr_text.split(",")]
        if len(parts) >= 2:
            # 尝试从最后一段获取国家
            last = parts[-1].strip()
            country_info = _COUNTRY_ALIAS.get(last)
            if country_info:
                result["country_code"] = country_info[0]
                result["currency"] = country_info[1]
                parts = parts[:-1]

            # 尝试提取邮编 (支持英式如 N2 8EY、美式如 90001、日式如 150-0002)
            for idx, p in enumerate(parts):
                if _re.search(r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b', p.strip(), _re.IGNORECASE):
                    result["postal_code"] = p.strip()
                    parts.pop(idx)
                    break
                elif _re.search(r'\b\d{5}(-\d{4})?\b', p.strip()):
                    result["postal_code"] = p.strip()
                    parts.pop(idx)
                    break
                elif _re.search(r'\b\d{3}-\d{4}\b', p.strip()):
                    result["postal_code"] = p.strip()
                    parts.pop(idx)
                    break

            # 解析: 第一段=街道, 第二段=城市(州/省), 后续合并
            if len(parts) == 1:
                result["address_line1"] = parts[0]
            elif len(parts) == 2:
                result["address_line1"] = parts[0]
                result["address_state"] = parts[1]
            elif len(parts) >= 3:
                # 例: Langley House, London, England → line1=Langley House, state=London
                result["address_line1"] = parts[0]
                result["address_state"] = parts[1]  # 城市作为州/省

    return result


# 国家 → (code, currency, state, address, postal_code)
COUNTRY_MAP = {
    "US - 美国": ("US", "USD", "California", "123 Main St", "90001"),
    "DE - 德国": ("DE", "EUR", "Berlin", "Hauptstraße 1", "10115"),
    "JP - 日本": ("JP", "JPY", "Tokyo", "1-1-1 Shibuya", "150-0002"),
    "GB - 英国": ("GB", "GBP", "London", "10 Downing St", "SW1A 2AA"),
    "FR - 法国": ("FR", "EUR", "Paris", "1 Rue de Rivoli", "75001"),
    "SG - 新加坡": ("SG", "SGD", "Singapore", "1 Raffles Place", "048616"),
    "HK - 香港": ("HK", "HKD", "Hong Kong", "1 Queen's Road", "000000"),
    "KR - 韩国": ("KR", "KRW", "Seoul", "1 Gangnam-daero", "06000"),
    "AU - 澳大利亚": ("AU", "AUD", "NSW", "1 George St", "2000"),
    "CA - 加拿大": ("CA", "CAD", "Ontario", "123 King St", "M5H 1A1"),
    "NL - 荷兰": ("NL", "EUR", "Amsterdam", "Damrak 1", "1012 LG"),
    "IT - 意大利": ("IT", "EUR", "Rome", "Via Roma 1", "00100"),
    "ES - 西班牙": ("ES", "EUR", "Madrid", "Calle Mayor 1", "28013"),
    "CH - 瑞士": ("CH", "CHF", "Zurich", "Bahnhofstrasse 1", "8001"),
}

st.set_page_config(page_title="Auto BindCard", page_icon="💳", layout="wide")

# ── CSS: 流水线 + 闪烁动画 ──
st.markdown("""
<style>
    .block-container { max-width: 1100px; padding-top: 1.5rem; }

    @keyframes blink {
        0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(243,156,18,0.6); }
        50% { opacity: 0.5; box-shadow: 0 0 20px rgba(243,156,18,0.9); }
    }

    .pipeline-wrap { display: flex; align-items: center; gap: 0; margin: 16px 0; }

    .pipeline-node {
        text-align: center; padding: 10px 14px; border-radius: 10px; min-width: 90px;
        border: 2px solid #555; background: rgba(30,30,30,0.6); flex-shrink: 0;
    }
    .pipeline-node .icon { font-size: 22px; }
    .pipeline-node .label { font-size: 12px; font-weight: 600; margin-top: 3px; }

    .pipeline-node.done   { border-color: #2ecc71; }
    .pipeline-node.done .label { color: #2ecc71; }
    .pipeline-node.running { border-color: #f39c12; animation: blink 1s ease-in-out infinite; }
    .pipeline-node.running .label { color: #f39c12; }
    .pipeline-node.error  { border-color: #e74c3c; }
    .pipeline-node.error .label { color: #e74c3c; }
    .pipeline-node.pending { border-color: #555; }
    .pipeline-node.pending .label { color: #888; }

    .pipeline-line {
        height: 2px; flex: 1; min-width: 20px;
        background: linear-gradient(90deg, #555, #555);
    }
    .pipeline-line.done { background: linear-gradient(90deg, #2ecc71, #2ecc71); }
    .pipeline-line.active { background: linear-gradient(90deg, #2ecc71, #f39c12); }
</style>
""", unsafe_allow_html=True)

ICONS = {"done": "✅", "running": "⏳", "error": "❌", "pending": "⬜"}

# 后台日志缓存（线程安全）。
_LOG_CACHE = deque(maxlen=5000)
_LOG_LOCK = threading.Lock()


def render_pipeline_html(nodes):
    """渲染连线式流水线"""
    parts = []
    prev_done = False
    for i, (name, status) in enumerate(nodes):
        if i > 0:
            if prev_done and status == "running":
                line_cls = "active"
            elif prev_done:
                line_cls = "done"
            else:
                line_cls = ""
            parts.append(f'<div class="pipeline-line {line_cls}"></div>')
        parts.append(
            f'<div class="pipeline-node {status}">'
            f'<div class="icon">{ICONS.get(status, "⬜")}</div>'
            f'<div class="label">{name}</div>'
            f'</div>'
        )
        prev_done = status == "done"
    return f'<div class="pipeline-wrap">{"".join(parts)}</div>'


# ── 日志 ──
class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        # 不在日志线程里访问 st.session_state，避免 ScriptRunContext 警告。
        msg = self.format(record)
        with _LOG_LOCK:
            _LOG_CACHE.append(msg)


def pull_captured_logs():
    """将后台日志搬运到 session_state，需在主线程调用。"""
    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = []
    with _LOG_LOCK:
        if not _LOG_CACHE:
            return
        st.session_state.log_buffer.extend(list(_LOG_CACHE))
        _LOG_CACHE.clear()


def clear_captured_logs():
    with _LOG_LOCK:
        _LOG_CACHE.clear()


def init_logging():
    handler = LogCapture()
    handler.setLevel(logging.INFO)
    handler._is_log_capture = True  # 标记
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # 按标记移除旧的 LogCapture (class 名/属性)
    root.handlers = [h for h in root.handlers if not getattr(h, '_is_log_capture', False)]
    root.addHandler(handler)
    # 过滤第三方噪音日志
    logging.getLogger("watchdog").setLevel(logging.WARNING)


for k, v in {"log_buffer": [], "running": False, "result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 每次 rerun 先同步一次日志缓存
pull_captured_logs()


# ════════════════════════════════════════
# 顶部
# ════════════════════════════════════════
st.title("💳 Auto BindCard")

col_step1, col_step2, col_step3, col_proxy = st.columns([1, 1, 1, 2])
with col_step1:
    do_register = st.checkbox("注册账号", value=True)
with col_step2:
    do_checkout = st.checkbox("创建 Checkout", value=True)
with col_step3:
    do_payment = st.checkbox("提交支付", value=True)
with col_proxy:
    proxy = st.text_input("代理 (可选)", placeholder="socks5://127.0.0.1:1080", label_visibility="collapsed")

# 打码服务配置
captcha_col1, captcha_col2 = st.columns([3, 1])
with captcha_col1:
    captcha_key = st.text_input("🔑 YesCaptcha API Key", value="27e2aa9da9a236b2a6cfcc3fa0f045fdec2a3633104361", type="password", help="用于解决 Stripe hCaptcha 挑战验证")
with captcha_col2:
    captcha_api_url = st.text_input("打码 API", value="https://api.yescaptcha.com")

st.divider()

# ════════════════════════════════════════
# 配置区
# ════════════════════════════════════════
cfg_col1, cfg_col2 = st.columns(2)

with cfg_col1:
    with st.expander("📧 邮箱 & Team Plan", expanded=True):
        mail_worker = st.text_input("邮箱 Worker", value="https://apimail.mkai.de5.net")
        mc1, mc2 = st.columns(2)
        mail_domain = mc1.text_input("邮箱域名", value="mkai.de5.net")
        mail_token = mc2.text_input("邮箱 Token", value="ma123999", type="password")
        st.markdown("---")
        tc1, tc2, tc3 = st.columns(3)
        workspace_name = tc1.text_input("Workspace", value="Artizancloud")
        seat_quantity = tc2.number_input("席位数", min_value=2, max_value=50, value=5)
        promo_campaign = tc3.text_input("活动 ID", value="team0dollar")

with cfg_col2:
    with st.expander("💰 账单地址", expanded=True):
        # 如果有解析出的国家，自动选择对应国家
        _parsed_cc = st.session_state.get("_parsed_country_code", "")
        _default_country_idx = 0
        if _parsed_cc:
            for i, label in enumerate(COUNTRY_MAP.keys()):
                if label.startswith(_parsed_cc):
                    _default_country_idx = i
                    break
        country_label = st.selectbox("国家", list(COUNTRY_MAP.keys()), index=_default_country_idx)
        country_code, default_currency, default_state, default_addr, default_zip = COUNTRY_MAP[country_label]
        # 覆盖: 如果解析出了值，优先使用
        _p_addr = st.session_state.get("_parsed_address_line1", "")
        _p_state = st.session_state.get("_parsed_address_state", "")
        _p_zip = st.session_state.get("_parsed_postal_code", "")
        _p_currency = st.session_state.get("_parsed_currency", "")
        bc1, bc2 = st.columns(2)
        billing_name = bc1.text_input("姓名", value="Test User")
        currency = bc2.text_input("货币", value=_p_currency or default_currency)
        bc3, bc4, bc5 = st.columns(3)
        address_line1 = bc3.text_input("地址", value=_p_addr or default_addr)
        address_state = bc4.text_input("州/省", value=_p_state or default_state)
        postal_code = bc5.text_input("邮编", value=_p_zip or default_zip)

if do_payment:
    with st.expander("粘贴卡片信息 (自动识别)", expanded=False):
        paste_text = st.text_area(
            "粘贴卡片/账单文本",
            height=150,
            placeholder="粘贴包含卡号、有效期、CVV、账单地址的文本，自动识别填充...\n\n例:\n4462 2200 0462 4356\n03/29\nCVV 173\n账单地址\nLangley House, London, England, N2 8EY, UK",
            key="paste_card_text",
        )
        if paste_text and st.button("🔍 识别并填充", key="parse_btn"):
            parsed = _parse_card_text(paste_text)
            if parsed.get("card_number"):
                st.session_state["_test_card_number"] = parsed["card_number"]
            if parsed.get("exp_month"):
                st.session_state["_test_exp_month"] = parsed["exp_month"]
            if parsed.get("exp_year"):
                st.session_state["_test_exp_year"] = parsed["exp_year"]
            if parsed.get("cvv"):
                st.session_state["_test_cvc"] = parsed["cvv"]
            if parsed.get("address_line1"):
                st.session_state["_parsed_address_line1"] = parsed["address_line1"]
            if parsed.get("address_state"):
                st.session_state["_parsed_address_state"] = parsed["address_state"]
            if parsed.get("postal_code"):
                st.session_state["_parsed_postal_code"] = parsed["postal_code"]
            if parsed.get("country_code"):
                st.session_state["_parsed_country_code"] = parsed["country_code"]
                st.session_state["_parsed_currency"] = parsed.get("currency", "")
            # 展示识别结果
            filled = []
            if parsed.get("card_number"):
                filled.append(f"卡号: {parsed['card_number'][:4]} **** **** {parsed['card_number'][-4:]}")
            if parsed.get("exp_month"):
                filled.append(f"有效期: {parsed['exp_month']}/{parsed['exp_year']}")
            if parsed.get("cvv"):
                filled.append(f"CVV: ***")
            if parsed.get("raw_address"):
                filled.append(f"地址: {parsed['raw_address']}")
            if parsed.get("country_code"):
                filled.append(f"国家: {parsed['country_code']}")
            if parsed.get("postal_code"):
                filled.append(f"邮编: {parsed['postal_code']}")
            if filled:
                st.success("✅ 已识别: " + " | ".join(filled))
            else:
                st.warning("未能识别卡片信息，请检查文本格式")
            st.rerun()

    with st.expander(" 信用卡 ⚠️ Live 模式 - 真实扣款", expanded=True):
        TEST_CARDS = {
            "4242 4242 4242 4242 (Visa 标准)": ("4242424242424242", "123"),
            "4000 0000 0000 0002 (Visa 被拒)": ("4000000000000002", "123"),
            "4000 0000 0000 0069 (Visa 过期)": ("4000000000000069", "123"),
            "4000 0000 0000 9995 (Visa 余额不足)": ("4000000000009995", "123"),
            "5555 5555 5555 4444 (Mastercard)": ("5555555555554444", "123"),
            "5200 8282 8282 8210 (MC Debit)": ("5200828282828210", "123"),
            "2223 0031 2200 3222 (MC 2系列)": ("2223003122003222", "123"),
            "3782 822463 10005 (Amex)": ("378282246310005", "1234"),
        }
        tc_sel = st.selectbox("🧪 快速填充测试卡", ["不填充"] + list(TEST_CARDS.keys()), key="tc_sel")
        if tc_sel != "不填充":
            tc_num, tc_cvc = TEST_CARDS[tc_sel]
            st.session_state["_test_card_number"] = tc_num
            st.session_state["_test_cvc"] = tc_cvc

        _tn = st.session_state.get("_test_card_number", "")
        _tm = st.session_state.get("_test_exp_month", "12")
        _ty = st.session_state.get("_test_exp_year", "2030")
        _tc = st.session_state.get("_test_cvc", "")

        cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 1])
        card_number = cc1.text_input("卡号", value=_tn, placeholder="真实卡号")
        exp_month = cc2.text_input("月", value=_tm)
        exp_year = cc3.text_input("年", value=_ty)
        card_cvc = cc4.text_input("CVC", value=_tc, type="password")

        if _tn and _tn.startswith("4"):
            st.caption("⚠️ Live 模式下所有测试卡都会被拒绝，仅用于验证流程")

# 跳过注册时可复用已有凭证
use_existing_creds = st.checkbox("🔐 使用已有凭证（跳过注册）", value=not do_register)
cred_email = ""
cred_session_token = ""
cred_access_token = ""
cred_device_id = ""
if use_existing_creds:
    with st.expander("🔐 已有凭证配置", expanded=not do_register):
        cred_files = []
        if os.path.exists(OUTPUT_DIR):
            cred_files = sorted(
                [f for f in os.listdir(OUTPUT_DIR) if f.startswith("credentials_") and f.endswith(".json")],
                reverse=True,
            )

        selected_cred = st.selectbox("凭证文件", ["手动输入"] + cred_files, index=1 if cred_files else 0)
        loaded = {}
        if selected_cred != "手动输入":
            try:
                with open(os.path.join(OUTPUT_DIR, selected_cred), "r") as f:
                    loaded = json.load(f)
            except Exception:
                loaded = {}

        cred_email = st.text_input("邮箱", value=loaded.get("email", ""))
        cred_session_token = st.text_input("session_token", value=loaded.get("session_token", ""), type="password")
        cred_access_token = st.text_input("access_token", value=loaded.get("access_token", ""), type="password")
        cred_device_id = st.text_input("device_id", value=loaded.get("device_id", ""))

st.divider()

# ════════════════════════════════════════
# Tab
# ════════════════════════════════════════
steps_list = []
if do_register: steps_list.append("注册")
if do_checkout: steps_list.append("Checkout")
if do_payment: steps_list.append("支付")

tab_run, tab_accounts, tab_history = st.tabs(["▶ 执行", "📋 账号", "📊 历史"])

# ── 构建节点 ──
ALL_NODES = [
    ("邮箱创建", "mail"),
    ("账号注册", "register"),
    ("Checkout", "checkout"),
    ("指纹获取", "fingerprint"),
    ("卡片Token", "tokenize"),
    ("确认支付", "confirm"),
    ("挑战验证", "challenge"),
]

def get_active_nodes():
    active = []
    if do_register:
        active += [("邮箱创建", "mail"), ("账号注册", "register")]
    if do_checkout:
        active += [("Checkout", "checkout"), ("指纹获取", "fingerprint")]
    if do_payment:
        active += [("卡片Token", "tokenize"), ("确认支付", "confirm"), ("挑战验证", "challenge")]
    return active


with tab_run:
    bc1, bc2 = st.columns([3, 1])
    with bc1:
        run_btn = st.button("🚀 开始执行", disabled=st.session_state.running or not steps_list, width="stretch", type="primary")
    with bc2:
        if st.button("🗑️ 清空", width="stretch"):
            st.session_state.log_buffer = []
            st.session_state.result = None
            clear_captured_logs()
            st.rerun()

    active_nodes = get_active_nodes()

    if run_btn:
        st.session_state.running = True
        st.session_state.log_buffer = []
        st.session_state.result = None
        clear_captured_logs()
        init_logging()

        pipeline_area = st.empty()
        log_area = st.empty()

        store = ResultStore(output_dir=OUTPUT_DIR)
        rd = {"success": False, "error": "", "email": "", "steps": {}}
        node_status = {key: "pending" for _, key in active_nodes}

        def _refresh():
            pipeline_area.markdown(
                render_pipeline_html([(name, node_status.get(key, "pending")) for name, key in active_nodes]),
                unsafe_allow_html=True,
            )

        _refresh()

        try:
            cfg = Config()
            cfg.proxy = proxy or None
            cfg.mail.email_domain = mail_domain
            cfg.mail.worker_domain = mail_worker
            cfg.mail.admin_token = mail_token
            cfg.team_plan.workspace_name = workspace_name
            cfg.team_plan.seat_quantity = seat_quantity
            cfg.team_plan.promo_campaign_id = promo_campaign
            cfg.captcha = CaptchaConfig(api_url=captcha_api_url, client_key=captcha_key)
            cfg.billing = BillingInfo(name=billing_name, email="", country=country_code, currency=currency,
                                      address_line1=address_line1, address_state=address_state,
                                      postal_code=postal_code)
            if do_payment:
                cfg.card = CardInfo(number=card_number, cvc=card_cvc, exp_month=exp_month, exp_year=exp_year)

            auth_result = None
            af = None

            # ── 注册/凭证 ──
            if do_register and not use_existing_creds:
                node_status["mail"] = "running"; _refresh()
                mp = MailProvider(worker_domain=cfg.mail.worker_domain, admin_token=cfg.mail.admin_token, email_domain=cfg.mail.email_domain)
                node_status["mail"] = "done"
                node_status["register"] = "running"; _refresh()
                af = AuthFlow(cfg)
                auth_result = af.run_register(mp)
                rd["email"] = auth_result.email
                rd["steps"]["mail"] = "✅"
                rd["steps"]["register"] = "✅"
                node_status["register"] = "done"; _refresh()
                store.save_credentials(auth_result.to_dict())
                store.append_credentials_csv(auth_result.to_dict())
                pull_captured_logs()
                log_area.code("\n".join(st.session_state.log_buffer[-60:]), language="log")
            elif use_existing_creds and do_checkout:
                # 跳过注册，直接使用已有凭证
                if not cred_session_token or not cred_access_token:
                    raise RuntimeError("跳过注册时必须提供 session_token 和 access_token")
                af = AuthFlow(cfg)
                auth_result = af.from_existing_credentials(
                    session_token=cred_session_token,
                    access_token=cred_access_token,
                    device_id=cred_device_id,
                )
                auth_result.email = cred_email or "unknown@example.com"
                rd["email"] = auth_result.email
                rd["steps"]["register"] = "⏭️"

            # ── Checkout ──
            if do_checkout:
                if not auth_result:
                    raise RuntimeError("需先注册或提供凭证")
                node_status["checkout"] = "running"; _refresh()
                cfg.billing.email = auth_result.email
                pf = PaymentFlow(cfg, auth_result)
                if af:
                    pf.session = af.session
                cs_id = pf.create_checkout_session()
                rd["checkout_session_id"] = cs_id
                rd["checkout_data"] = pf.checkout_data
                rd["steps"]["checkout"] = "✅"
                node_status["checkout"] = "done"; _refresh()
                pull_captured_logs()
                log_area.code("\n".join(st.session_state.log_buffer[-60:]), language="log")

                node_status["fingerprint"] = "running"; _refresh()
                pf.fetch_stripe_fingerprint()
                pf.extract_stripe_pk(pf.checkout_url)
                rd["stripe_pk"] = (pf.stripe_pk[:30] + "...") if pf.stripe_pk else ""
                rd["steps"]["fingerprint"] = "✅"
                node_status["fingerprint"] = "done"; _refresh()
                pull_captured_logs()
                log_area.code("\n".join(st.session_state.log_buffer[-60:]), language="log")

                # ── 支付 ──
                if do_payment:
                    node_status["tokenize"] = "running"; _refresh()
                    pf.payment_method_id = pf.create_payment_method()
                    rd["steps"]["tokenize"] = "✅"
                    node_status["tokenize"] = "done"; _refresh()
                    pull_captured_logs()
                    log_area.code("\n".join(st.session_state.log_buffer[-60:]), language="log")

                    node_status["confirm"] = "running"; _refresh()
                    pf.fetch_payment_page_details(cs_id)
                    pay = pf.confirm_payment(cs_id)
                    rd["confirm_status"] = pay.confirm_status
                    rd["confirm_response"] = pay.confirm_response
                    pull_captured_logs()
                    log_area.code("\n".join(st.session_state.log_buffer[-60:]), language="log")

                    # confirm 本身完成，更新节点状态
                    # challenge 验证结果已经包含在 pay 里了 (confirm_payment 内部处理)
                    if pay.success:
                        rd["steps"]["confirm"] = "✅"
                        rd["steps"]["challenge"] = "✅"
                        node_status["confirm"] = "done"
                        node_status["challenge"] = "done"
                    elif pay.error and "hCaptcha" in pay.error:
                        rd["steps"]["confirm"] = "✅"  # confirm 本身成功了
                        rd["steps"]["challenge"] = "❌"
                        node_status["confirm"] = "done"
                        node_status["challenge"] = "error"
                    elif pay.error and "requires_action" in pay.error:
                        rd["steps"]["confirm"] = "✅"
                        rd["steps"]["challenge"] = "❌"
                        node_status["confirm"] = "done"
                        node_status["challenge"] = "error"
                    else:
                        rd["steps"]["confirm"] = "❌"
                        rd["steps"]["challenge"] = "⏭️"
                        node_status["confirm"] = "error"
                        node_status["challenge"] = "pending"

                    rd["success"] = pay.success
                    rd["error"] = pay.error
                    _refresh()
                    pull_captured_logs()
                    log_area.code("\n".join(st.session_state.log_buffer[-60:]), language="log")
                else:
                    rd["success"] = True
            elif do_register:
                rd["success"] = True

        except Exception as e:
            rd["error"] = str(e)
            st.session_state.log_buffer.append(f"EXCEPTION:\n{traceback.format_exc()}")
            # 尝试提取 Stripe 原始响应
            try:
                if 'pf' in dir() and pf and pf.result.confirm_response:
                    rd["confirm_response"] = pf.result.confirm_response
                    rd["confirm_status"] = pf.result.confirm_status
            except Exception:
                pass
            for k in node_status:
                if node_status[k] == "running":
                    node_status[k] = "error"
            _refresh()

        st.session_state.result = rd
        st.session_state.running = False

        try:
            store.save_result(rd, "ui_run")
            if rd.get("email"):
                store.append_history(email=rd["email"], status="ui_run",
                                     checkout_session_id=rd.get("checkout_session_id", ""),
                                     payment_status=rd.get("confirm_status", ""),
                                     error=rd.get("error", ""))
        except Exception:
            pass

        # 最终结果
        if rd["success"]:
            st.success(f"✅ 全部完成! {rd.get('email', '')}")
        else:
            st.error(f"❌ {rd.get('error', '')}")

        # 如果有 Stripe 原始返回，展示
        if rd.get("confirm_response"):
            with st.expander("Stripe 原始响应", expanded=True):
                st.json(rd["confirm_response"])

        # 如果有 checkout 返回，展示
        if rd.get("checkout_data"):
            with st.expander("ChatGPT Checkout 响应 (含 discount 信息)", expanded=False):
                st.json(rd["checkout_data"])

        pull_captured_logs()
        log_area.code("\n".join(st.session_state.log_buffer[-200:]), language="log")

    elif st.session_state.log_buffer:
        pull_captured_logs()
        st.code("\n".join(st.session_state.log_buffer[-200:]), language="log")

    # ── 结果 ──
    if st.session_state.result and not run_btn:
        r = st.session_state.result

        if r.get("steps"):
            node_status_saved = {}
            for _, key in active_nodes:
                node_status_saved[key] = "done" if r["steps"].get(key, "").startswith("✅") else ("error" if key in r["steps"] else "pending")
            st.markdown(
                render_pipeline_html([(name, node_status_saved.get(key, "pending")) for name, key in active_nodes]),
                unsafe_allow_html=True,
            )

        st.divider()
        cols = st.columns(4)
        cols[0].metric("邮箱", r.get("email") or "-")
        cols[1].metric("Checkout", (r.get("checkout_session_id", "")[:20] + "...") if r.get("checkout_session_id") else "-")
        cols[2].metric("Confirm", r.get("confirm_status") or "-")
        cols[3].metric("状态", "成功" if r.get("success") else "失败")

        if r.get("confirm_response"):
            with st.expander("Stripe 原始响应", expanded=False):
                st.json(r["confirm_response"])

        with st.expander("完整 JSON 结果", expanded=False):
            st.json(r)


# ════════════════════════════════════════
# Tab: 账号
# ════════════════════════════════════════
with tab_accounts:
    csv_path = os.path.join(OUTPUT_DIR, "accounts.csv")
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if not df.empty:
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption(f"共 {len(df)} 条记录")
                if st.button("🔄 刷新", key="ref_acc"):
                    st.rerun()
            else:
                st.info("暂无账号记录")
        except Exception as e:
            st.error(str(e))
    else:
        st.info("暂无账号。注册后自动保存到此处。")

    st.divider()
    with st.expander("📁 凭证文件", expanded=False):
        if os.path.exists(OUTPUT_DIR):
            cred_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("credentials_") and f.endswith(".json")], reverse=True)
            if cred_files:
                sel = st.selectbox("选择凭证文件", cred_files, key="cred_sel")
                if sel:
                    with open(os.path.join(OUTPUT_DIR, sel)) as f:
                        data = json.load(f)
                    st.json({k: (v[:50] + "..." + v[-20:] if isinstance(v, str) and len(v) > 80 else v) for k, v in data.items()})
            else:
                st.caption("暂无凭证文件")


# ════════════════════════════════════════
# Tab: 历史
# ════════════════════════════════════════
with tab_history:
    hist_path = os.path.join(OUTPUT_DIR, "history.csv")
    if os.path.exists(hist_path):
        try:
            import pandas as pd
            df = pd.read_csv(hist_path)
            if not df.empty:
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption(f"共 {len(df)} 条")
                if st.button("🔄 刷新", key="ref_hist"):
                    st.rerun()
            else:
                st.info("暂无历史")
        except Exception as e:
            st.error(str(e))
    else:
        st.info("暂无执行历史")

    st.divider()
    with st.expander("📁 结果文件", expanded=False):
        if os.path.exists(OUTPUT_DIR):
            rf = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json") and not f.startswith("credentials_") and not f.startswith("debug_")], reverse=True)
            if rf:
                sel = st.selectbox("选择结果文件", rf, key="res_sel")
                if sel:
                    with open(os.path.join(OUTPUT_DIR, sel)) as f:
                        st.json(json.load(f))
            else:
                st.caption("暂无结果文件")

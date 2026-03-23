"""
自动化绑卡支付 - 主入口
用法:
  1. 全流程（注册 + 支付）:
     python main.py --config config.json

  2. 仅支付（已有凭证）:
     python main.py --config config.json --skip-register

  3. 交互式输入卡信息:
     python main.py --config config.json --interactive
"""
import argparse
import base64
from collections import Counter
import json
import logging
import os
import sys
import time
from datetime import datetime

from config import Config, CardInfo
from mail_provider import MailProvider
from auth_flow import AuthFlow, AuthResult
from payment_flow import PaymentFlow
from logger import setup_logging, ResultStore

logger = logging.getLogger("main")


def _b64url_decode(data: str) -> bytes:
    data += "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def extract_chatgpt_plan_type(access_token: str) -> str:
    """从 access_token 的 JWT payload 提取 chatgpt_plan_type。"""
    if not access_token or access_token.count(".") < 2:
        return "unknown"
    try:
        payload_b64 = access_token.split(".")[1]
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        auth_claim = payload.get("https://api.openai.com/auth", {})
        return auth_claim.get("chatgpt_plan_type", "unknown")
    except Exception:
        return "unknown"


def refresh_access_token_for_plan(auth_flow: AuthFlow, fallback_token: str) -> str:
    """支付后刷新 access_token，尽量拿到最新计划类型。"""
    headers = auth_flow._common_headers("https://chatgpt.com/")
    for i in range(2):
        try:
            resp = auth_flow.session.get(
                "https://chatgpt.com/api/auth/session",
                headers=headers,
                timeout=30,
            )
            token = resp.json().get("accessToken", "")
            if token:
                return token
        except Exception as e:
            logger.warning(f"支付后刷新 access_token 失败(第{i+1}次): {e}")
        if i == 0:
            time.sleep(3)
    return fallback_token


def determine_competition_status(plan_type: str, payment_result) -> tuple[bool, str]:
    """
    严格判定：
    - 仅当 access_token 中的 plan_type 非 free/unknown 时视为成功。
    - 不做软判定，不接受基于 confirm 2xx 的兜底通过。
    """
    if plan_type not in ("free", "unknown"):
        return True, ""

    original = (payment_result.error or "").strip()
    if original:
        return False, f"{original};plan_not_upgraded:{plan_type}"
    if payment_result.error == "requires_confirmation_stage":
        return False, "requires_confirmation_stage"
    return False, f"plan_not_upgraded:{plan_type}"


def classify_failure_reason(err: str) -> str:
    """将失败原因归类，便于自动重试统计。"""
    e = (err or "").lower()
    if not e:
        return "unknown"
    if "registration_disallowed" in e:
        return "registration_disallowed"
    if "create_account" in e and "400" in e:
        return "registration_failed"
    if "setup_intent_authentication_failure" in e or "payment_intent_authentication_failure" in e:
        return "captcha_auth_failed"
    if "error_zero_balance" in e or "余额不足" in e:
        return "captcha_no_balance"
    if "captcha_unsolved_or_provider_error" in e:
        return "captcha_unsolved"
    if "requires_3ds" in e or "redirect" in e:
        return "requires_3ds"
    if "token_invalidated" in e:
        return "token_invalidated"
    if "plan_not_upgraded" in e:
        return "plan_not_upgraded"
    if "hcaptcha" in e:
        return "hcaptcha_related"
    if "timeout" in e:
        return "timeout"
    return "other"


def interactive_card_input() -> CardInfo:
    """交互式输入卡信息"""
    print("\n=== 请输入信用卡信息 ===")
    card = CardInfo()
    card.number = input("卡号: ").strip().replace(" ", "")
    card.exp_month = input("到期月份 (MM): ").strip()
    card.exp_year = input("到期年份 (YY or YYYY): ").strip()
    card.cvc = input("CVC: ").strip()
    return card


def save_result(result: dict, prefix: str = "result"):
    """保存结果到文件（兼容直接调用）"""
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/{prefix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {path}")
    return path


def run_full_flow(config: Config, skip_register: bool = False):
    """执行完整流程"""

    store = ResultStore()

    final_result = {
        "timestamp": datetime.now().isoformat(),
        "auth": {},
        "payment": {},
    }

    # ── 阶段 1: 注册/登录 ──
    auth_flow = AuthFlow(config)

    if skip_register:
        if not (config.session_token and config.access_token):
            logger.error("跳过注册模式需要提供 session_token 和 access_token")
            sys.exit(1)
        auth_result = auth_flow.from_existing_credentials(
            session_token=config.session_token,
            access_token=config.access_token,
            device_id=config.device_id or "",
        )
        logger.info("使用已有凭证，跳过注册")
    else:
        mail = MailProvider(
            imap_server=config.mail.imap_server,
            imap_port=config.mail.imap_port,
            email_addr=config.mail.email,
            auth_code=config.mail.auth_code,
            catch_all_domain=config.mail.catch_all_domain,
        )
        auth_result = auth_flow.run_register(mail)
        logger.info(f"注册成功: {auth_result.email}")
        # 保存凭证到 JSON 和 CSV
        cred_path = store.save_credentials(auth_result.to_dict())
        csv_path = store.append_credentials_csv(auth_result.to_dict())
        logger.info(f"凭证已保存: {cred_path}")
        logger.info(f"凭证已追加到: {csv_path}")

    final_result["auth"] = auth_result.to_dict()

    # ── 阶段 2: 支付 ──
    if not config.card.number:
        logger.error("未配置信用卡信息，无法执行支付")
        path = store.save_result(final_result, "register_only")
        store.append_history(
            email=auth_result.email,
            status="register_only",
            detail_file=path,
        )
        return final_result

    # 如果 billing email 为空，使用注册邮箱
    if not config.billing.email:
        config.billing.email = auth_result.email

    payment_flow = PaymentFlow(config, auth_result, stripe_proxy=config.proxy)
    payment_result = payment_flow.run_payment()

    # ── 赛题判定：用最新 access_token 的 plan_type 判断是否升级成功 ──
    latest_access_token = refresh_access_token_for_plan(auth_flow, auth_result.access_token)
    plan_type = extract_chatgpt_plan_type(latest_access_token)
    final_result["plan_type"] = plan_type

    # 只有计划类型升级到非 free 才视为赛题成功
    ok, reason = determine_competition_status(plan_type, payment_result)
    payment_result.success = ok
    payment_result.error = reason

    logger.info(
        "exp_tag=%s",
        json.dumps(
            {
                **(payment_result.experiment_tag or {}),
                "stage": "final_summary",
                "plan_type": plan_type,
                "payment_success": ok,
                "confirm_status": payment_result.confirm_status,
                "reason": reason,
                "checkout_session_id": (payment_result.checkout_session_id[:24] if payment_result.checkout_session_id else ""),
            },
            ensure_ascii=False,
        ),
    )

    final_result["payment"] = payment_result.to_dict()

    # ── 保存结果 ──
    prefix = "success" if payment_result.success else "failed"
    path = store.save_result(final_result, prefix)

    # ── 追加历史 ──
    store.append_history(
        email=auth_result.email,
        status=prefix,
        checkout_session_id=payment_result.checkout_session_id,
        payment_status=payment_result.confirm_status,
        error=payment_result.error,
        detail_file=path,
    )

    # ── 输出摘要 ──
    print("\n" + "=" * 60)
    if payment_result.success:
        print("✅ 绑卡支付成功!")
    elif payment_result.error == "requires_confirmation_stage":
        print("⚠️  已生成 OpenAI 订阅链接，但尚未完成最终验证确认阶段")
    elif payment_result.error == "requires_3ds_verification":
        print("⚠️  支付需要 3DS 验证，请手动完成")
    else:
        print(f"❌ 支付失败: {payment_result.error}")
    print(f"   邮箱: {auth_result.email}")
    print(f"   Checkout Session: {payment_result.checkout_session_id[:30]}...")
    if getattr(payment_result, "openai_checkout_url", ""):
        print(f"   OpenAI Checkout URL: {payment_result.openai_checkout_url}")
    if getattr(payment_result, "openai_client_secret", ""):
        print(f"   OpenAI Client Secret: {payment_result.openai_client_secret[:60]}...")
    print(f"   Plan Type: {final_result.get('plan_type', 'unknown')}")
    print("=" * 60)

    return final_result


def run_auto_retry(
    config_path: str,
    skip_register: bool = False,
    max_attempts: int = 5,
    retry_interval: int = 5,
    interactive_card: CardInfo = None,
):
    """
    自动重试模式：
    - 每轮默认新注册账号并支付（skip_register=False）
    - 直到成功或达到最大轮数
    """
    max_attempts = max(1, int(max_attempts))
    retry_interval = max(0, int(retry_interval))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "auto_retry",
        "max_attempts": max_attempts,
        "attempts": [],
        "success": False,
        "success_attempt": 0,
        "failure_stats": {},
    }
    reason_counter = Counter()
    deterministic_streak = 0
    try:
        deterministic_stop_after = max(1, int(os.getenv("AUTO_DETERMINISTIC_STOP_AFTER", "2")))
    except Exception:
        deterministic_stop_after = 2

    for idx in range(1, max_attempts + 1):
        logger.info("=" * 80)
        logger.info("[AUTO] 开始第 %s/%s 轮", idx, max_attempts)
        logger.info("=" * 80)

        # 每轮重新加载配置，避免状态污染
        if os.path.exists(config_path):
            cfg = Config.from_file(config_path)
        else:
            cfg = Config()

        if interactive_card:
            cfg.card = interactive_card

        attempt_start = time.time()
        result = {}
        is_ok = False
        reason = ""
        payment = {}
        try:
            result = run_full_flow(cfg, skip_register=skip_register)
            payment = result.get("payment", {}) if isinstance(result, dict) else {}
            is_ok = bool(payment.get("success"))
            reason = payment.get("error", "")
        except Exception as e:
            # auto-retry 模式下，不因单轮异常中断总流程
            reason = f"exception:{type(e).__name__}:{e}"
            logger.exception("[AUTO] 第 %s 轮异常，继续下一轮: %s", idx, reason)
            result = {
                "auth": {},
                "payment": {"success": False, "error": reason},
                "plan_type": "unknown",
            }
        duration = round(time.time() - attempt_start, 2)
        reason_cls = classify_failure_reason(reason)
        reason_counter[reason_cls] += 0 if is_ok else 1

        attempt_info = {
            "attempt": idx,
            "success": is_ok,
            "duration_sec": duration,
            "email": (result.get("auth", {}) or {}).get("email", ""),
            "checkout_session_id": payment.get("checkout_session_id", ""),
            "confirm_status": payment.get("confirm_status", ""),
            "plan_type": result.get("plan_type", "unknown"),
            "error": reason,
            "error_class": reason_cls,
        }
        summary["attempts"].append(attempt_info)

        if is_ok:
            summary["success"] = True
            summary["success_attempt"] = idx
            logger.info("[AUTO] 第 %s 轮成功，停止重试", idx)
            break

        # 避免盲目硬跑：同类确定性失败连续出现时提前停止
        is_deterministic_fail = (
            reason_cls == "captcha_auth_failed"
            and "verify_auth_failed:setup_intent_authentication_failure" in (reason or "")
            and str(payment.get("confirm_status", "") or "").startswith("200")
        )
        if is_deterministic_fail:
            deterministic_streak += 1
            logger.warning(
                "[AUTO] 检测到确定性失败 #%s: %s",
                deterministic_streak,
                "verify_auth_failed:setup_intent_authentication_failure",
            )
        else:
            deterministic_streak = 0

        if deterministic_streak >= deterministic_stop_after:
            logger.warning(
                "[AUTO] 连续 %s 轮确定性失败，停止重试以避免硬跑。请先调整策略后再试。",
                deterministic_streak,
            )
            break

        if idx < max_attempts and retry_interval > 0:
            logger.info("[AUTO] 第 %s 轮失败，%s 秒后重试", idx, retry_interval)
            time.sleep(retry_interval)

    summary["failure_stats"] = dict(reason_counter)

    out_path = save_result(summary, "auto_retry_summary")
    logger.info("[AUTO] 汇总已保存: %s", out_path)

    print("\n" + "=" * 60)
    if summary["success"]:
        print(f"✅ AUTO 模式成功: 第 {summary['success_attempt']}/{max_attempts} 轮命中")
    else:
        print(f"❌ AUTO 模式失败: 已尝试 {max_attempts} 轮，均未成功")
    print(f"   失败分类统计: {summary['failure_stats']}")
    print(f"   汇总文件: {out_path}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="自动化绑卡支付")
    parser.add_argument("--config", "-c", default="config.json", help="配置文件路径")
    parser.add_argument("--skip-register", action="store_true", help="跳过注册，使用已有凭证")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互式输入卡信息")
    parser.add_argument("--auto-retry", action="store_true", help="自动重试直到成功（建议配合新注册）")
    parser.add_argument("--max-attempts", type=int, default=1, help="自动重试最大轮数")
    parser.add_argument("--retry-interval", type=int, default=5, help="自动重试间隔秒")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")
    args = parser.parse_args()

    if args.debug:
        log_file = setup_logging(debug=True)
    else:
        log_file = setup_logging(debug=False)
    logger.info(f"日志文件: {log_file}")

    # 加载配置
    if os.path.exists(args.config):
        config = Config.from_file(args.config)
        logger.info(f"配置已加载: {args.config}")
    else:
        config = Config()
        logger.warning(f"配置文件 {args.config} 不存在，使用默认配置")

    # 交互式卡信息（一次输入，复用到 auto-retry 每轮）
    interactive_card = None
    if args.interactive:
        interactive_card = interactive_card_input()
        config.card = interactive_card

    # 自动重试模式
    if args.auto_retry or args.max_attempts > 1:
        if args.skip_register:
            logger.warning("AUTO + skip-register 容易因 token 失效导致连续失败，建议去掉 --skip-register")
        run_auto_retry(
            config_path=args.config,
            skip_register=args.skip_register,
            max_attempts=args.max_attempts,
            retry_interval=args.retry_interval,
            interactive_card=interactive_card,
        )
        return

    run_full_flow(config, skip_register=args.skip_register)


if __name__ == "__main__":
    main()

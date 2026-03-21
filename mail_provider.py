"""
邮箱服务 - 通过 IMAP 收取 OTP 验证码
使用 catch-all 域名生成随机邮箱，通过 IMAP 从 QQ 邮箱收取转发的验证码
"""
import email
import email.message
import imaplib
import random
import re
import string
import time
import logging
from email.header import decode_header

logger = logging.getLogger(__name__)


class MailProvider:
    """IMAP 邮箱提供者 (catch-all 域名 + QQ 邮箱 IMAP)"""

    def __init__(self, imap_server: str, imap_port: int, email_addr: str, auth_code: str,
                 catch_all_domain: str = ""):
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.email_addr = email_addr
        self.auth_code = auth_code
        self.catch_all_domain = catch_all_domain

    def _connect(self) -> imaplib.IMAP4_SSL:
        """建立 IMAP 连接并登录"""
        conn = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
        conn.login(self.email_addr, self.auth_code)
        return conn

    @staticmethod
    def _random_name() -> str:
        letters1 = "".join(random.choices(string.ascii_lowercase, k=5))
        numbers = "".join(random.choices(string.digits, k=random.randint(1, 3)))
        letters2 = "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 3)))
        return letters1 + numbers + letters2

    def create_mailbox(self) -> str:
        """生成随机 catch-all 邮箱地址，并验证 IMAP 连接"""
        conn = self._connect()
        conn.logout()

        if self.catch_all_domain:
            addr = f"{self._random_name()}@{self.catch_all_domain}"
        else:
            addr = self.email_addr

        logger.info(f"邮箱已创建: {addr} (IMAP 收件: {self.email_addr})")
        return addr

    @staticmethod
    def _decode_payload(msg: email.message.Message) -> str:
        """提取邮件正文"""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct in ("text/plain", "text/html"):
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body += payload.decode(charset, errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")
        return body

    @staticmethod
    def _extract_otp(content: str) -> str | None:
        """从邮件内容中提取 OTP"""
        patterns = [r"代码为\s*(\d{6})", r"code is\s*(\d{6})", r"(\d{6})"]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                return matches[0]
        return None

    def _match_recipient(self, msg: email.message.Message, target_email: str) -> bool:
        """检查邮件的收件人是否匹配目标地址"""
        for header in ("To", "Cc", "Delivered-To", "X-Original-To"):
            val = msg.get(header, "")
            if target_email.lower() in val.lower():
                return True
        return False

    def wait_for_otp(self, email_addr: str, timeout: int = 120) -> str:
        """阻塞等待 OTP 验证码"""
        logger.info(f"等待 OTP 验证码 -> {email_addr} (最长 {timeout}s)...")
        start = time.time()

        while time.time() - start < timeout:
            try:
                conn = self._connect()
                conn.select("INBOX")
                # 搜索来自 OpenAI 的未读邮件
                status, data = conn.search(None, '(UNSEEN FROM "openai")')
                if status == "OK" and data[0]:
                    mail_ids = data[0].split()
                    for mid in reversed(mail_ids):
                        status, msg_data = conn.fetch(mid, "(RFC822)")
                        if status != "OK":
                            continue
                        raw = msg_data[0][1]
                        msg = email.message_from_bytes(raw)
                        # 确认是发给目标地址的
                        if not self._match_recipient(msg, email_addr):
                            continue
                        body = self._decode_payload(msg)
                        otp = self._extract_otp(body)
                        if otp:
                            logger.info(f"收到 OTP: {otp}")
                            conn.store(mid, "+FLAGS", "\\Seen")
                            conn.logout()
                            return otp
                conn.logout()
            except Exception as e:
                logger.debug(f"IMAP 轮询异常: {e}")
            time.sleep(5)

        raise TimeoutError(f"等待 OTP 超时 ({timeout}s)")

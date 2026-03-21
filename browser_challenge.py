"""
Playwright 浏览器处理 Stripe hCaptcha 挑战

两种方式:
  1. solve() — 用 Stripe.js handleNextAction() 自动处理
  2. solve_hcaptcha_direct() — 直接加载 hCaptcha SDK, 获取 token, 用 API 提交

方式2更灵活: 在真实浏览器中执行 hCaptcha invisible challenge,
然后把 token 返回给 Python 代码通过 verify_challenge API 提交。
"""
import json
import logging
import math
import os
import re
import time
import base64
from urllib.parse import parse_qsl, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

try:
    from http_client import USER_AGENT as _DEFAULT_HTTP_UA
except Exception:
    _DEFAULT_HTTP_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    )


class BrowserChallengeSolver:
    """用真实浏览器处理 Stripe hCaptcha 挑战"""

    def __init__(self, stripe_pk: str, proxy: str = None, headless: bool = True):
        self.stripe_pk = stripe_pk
        self.proxy = proxy
        self.headless = headless
        self.user_agent = (os.getenv("BROWSER_CHALLENGE_UA") or _DEFAULT_HTTP_UA).strip()
        # 记录 challenge prompt 的尝试次数，避免每轮都用同一点击策略。
        self._challenge_attempt_state = {}
        # 记录每个题面的枚举进度（9-bit Gray code）。
        self._visual_enum_state = {}
        # 记录每个 challenge frame 最近一次可用 prompt，过渡态可复用。
        self._last_visual_prompt = {}
        self._last_checkbox_click_ts = 0.0
        # 拖拽题枚举状态（用于 AutoLLM 不稳定时逐点尝试）。
        self._drag_try_state = {}
        # 区域点选题（area_select）轮询状态。
        self._area_try_state = {}
        # 单选题轮询状态（避免重复点击同一候选）。
        self._single_try_state = {}
        # 最近一次 getcaptcha 的题面（用于 DOM 无 prompt 时回退）。
        self._last_net_prompt = ""
        self._last_getcaptcha_payload = {}
        self._last_getcaptcha_ts = 0.0
        # 记录最近一次 checkcaptcha 的 pass 状态，用于节流 verify_challenge。
        self._last_checkcaptcha_pass = None
        self._verify_blocked_count = 0
        # 最近一次 checkcaptcha 成功返回的 pass token / ekey（用于回退人工 verify）。
        self._last_generated_pass_uuid = ""
        self._last_checkcaptcha_ekey = ""
        self._last_verify_request_body = ""

    def _solve_pattern_with_llm(
        self,
        image_path: str,
        max_index: int,
        prompt: str = "",
        force_single: bool = False,
    ) -> list[int]:
        """
        使用本地/代理 OpenAI 兼容接口解 4x4「找异常」题。
        返回 1-based index 列表；失败返回 []。
        """
        if not image_path or max_index <= 0:
            return []

        if os.getenv("HCAPTCHA_AUTO_PATTERN_LLM", "0") in ("0", "false", "False"):
            return []

        api_key = (
            os.getenv("HCAPTCHA_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        if not api_key:
            logger.warning("[Browser][AutoLLM] 缺少 API Key，跳过自动识别")
            return []

        endpoint = (
            os.getenv("HCAPTCHA_LLM_BASE_URL")
            or "http://127.0.0.1:8317/v1/chat/completions"
        ).strip()
        model = (os.getenv("HCAPTCHA_LLM_MODEL") or "gpt-5.4-mini").strip()
        reasoning_effort = (os.getenv("HCAPTCHA_LLM_REASONING_EFFORT") or "low").strip().lower()
        if reasoning_effort not in ("low", "medium", "high", "xhigh"):
            reasoning_effort = "low"
        try:
            timeout_s = max(6, int(os.getenv("HCAPTCHA_LLM_TIMEOUT", "35")))
        except Exception:
            timeout_s = 35

        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            logger.warning("[Browser][AutoLLM] 读取图片失败: %s", e)
            return []

        prompt_norm = (prompt or "").strip()
        prompt_lc = prompt_norm.lower()
        expects_single = force_single or (
            "select the one" in prompt_lc
            or "identify the one" in prompt_lc
            or "matching silhouette" in prompt_lc
            or "line breaks" in prompt_lc
            or ("sample image" in prompt_lc and ("choose everything" in prompt_lc or "identify everything" in prompt_lc))
            or ("example image" in prompt_lc and ("choose everything" in prompt_lc or "identify everything" in prompt_lc))
            or "disrupts the pattern" in prompt_lc
            or "odd one out" in prompt_lc
            or "doesn't belong" in prompt_lc
            or "does not belong" in prompt_lc
            or "drag the segment" in prompt_lc
            or "complete the line" in prompt_lc
            or "drag the piece" in prompt_lc
            or "drag the object" in prompt_lc
            or "matching shadow" in prompt_lc
        )

        task_text = (
            "Solve the visual selection task shown in the image.\n"
            f"Task prompt: {prompt_norm or '(none)'}\n"
            f"The tiles are labeled 1-{max_index} in row-major order on the image.\n"
        )
        if expects_single:
            task_text += f"Return only ONE integer between 1 and {max_index}."
        else:
            task_text += (
                f"Return ONLY comma-separated tile indices between 1 and {max_index}. "
                "If none, return 0. Be conservative and avoid selecting too many uncertain tiles."
            )

        payload = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "temperature": 0,
            "max_tokens": 20,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": task_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                    ],
                }
            ],
        }

        try:
            import requests

            resp = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout_s,
            )
        except Exception as e:
            logger.warning("[Browser][AutoLLM] 请求异常: %s", e)
            return []

        if resp.status_code != 200:
            logger.warning("[Browser][AutoLLM] 请求失败: http=%s body=%s", resp.status_code, (resp.text or "")[:220])
            return []

        try:
            data = resp.json()
        except Exception:
            logger.warning("[Browser][AutoLLM] JSON 解析失败: %s", (resp.text or "")[:220])
            return []

        content = ""
        try:
            content = (
                ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            )
        except Exception:
            content = ""
        if not content:
            return []

        nums = re.findall(r"\b([1-9]|1[0-9]|2[0-9]|3[0-9])\b", str(content))
        if not nums:
            logger.warning("[Browser][AutoLLM] 未提取到有效索引: %s", str(content)[:200])
            return []

        picks: list[int] = []
        for n in nums:
            try:
                idx = int(n)
            except Exception:
                continue
            if 1 <= idx <= max_index and idx not in picks:
                picks.append(idx)

        if not picks:
            return []
        if expects_single:
            picks = picks[:1]
        else:
            try:
                max_picks = max(1, min(max_index, int(os.getenv("HCAPTCHA_AUTO_MAX_PICKS", "4"))))
            except Exception:
                max_picks = min(max_index, 4)
            if len(picks) > max_picks:
                picks = picks[:max_picks]
        logger.info("[Browser][AutoLLM] 识别索引: %s / max=%s prompt=%s", picks, max_index, prompt_norm[:120])
        return picks

    def solve_hcaptcha_direct(self, site_key: str, site_url: str, rqdata: str = "", timeout: int = 60) -> dict:
        """
        在真实浏览器中执行 hCaptcha invisible 挑战。
        直接加载 hCaptcha SDK, 执行 invisible challenge, 返回 token。
        使用增强反检测: 鼠标模拟 + stealth + 持久化 profile。

        返回:
            {"success": True, "token": "...", "ekey": "..."} 或
            {"success": False, "error": "..."}
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {"success": False, "error": "playwright not installed"}

        import os, random

        logger.info(f"[Browser] 直接 hCaptcha: site_key={site_key[:20]}...")
        logger.info(f"[Browser] headless={self.headless}, proxy={self.proxy or '直连'}")

        # 使用 Stripe 提供的 hosted URL 域作为挑战来源，避免 sitekey 域不匹配。
        # 注意: fragment (#...) 不会参与 HTTP 请求，需剥离后再路由/导航。
        target_url = (site_url or "").strip()
        parsed = urlsplit(target_url) if target_url else None
        if (not parsed) or (not parsed.scheme) or (not parsed.netloc):
            target_url = "https://checkout.stripe.com/challenge/verify"
            parsed = urlsplit(target_url)
        target_nav_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
        route_pattern = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}*"
        logger.info(f"[Browser] hCaptcha target_url={target_nav_url[:140]}")

        # hCaptcha 页面 HTML — 更真实的页面结构
        hcaptcha_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Secure Payment Verification</title>
    <script src="https://js.hcaptcha.com/1/api.js?render=explicit&onload=onHcaptchaLoad" async defer></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 40px; background: #f6f9fc; }
        .container { max-width: 480px; margin: 0 auto; background: white; border-radius: 8px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h2 { color: #32325d; margin: 0 0 16px 0; font-size: 20px; }
        p { color: #6b7c93; margin: 0 0 24px 0; line-height: 1.5; }
        #hcaptcha-container { min-height: 60px; }
        .footer { margin-top: 24px; text-align: center; color: #aab7c4; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Verifying your identity</h2>
        <p>Please wait while we verify your payment. This process is automatic and should complete shortly.</p>
        <div id="hcaptcha-container"></div>
        <div class="footer">Secured by Stripe</div>
    </div>
    <script>
        window.__hcaptchaResult = null;
        window.__hcaptchaError = null;
        window.__hcaptchaReady = false;
        function onHcaptchaLoad() { window.__hcaptchaReady = true; }
    </script>
</body>
</html>"""

        # 持久化浏览器 profile 目录
        profile_dir = os.path.join(os.path.dirname(__file__), ".browser_profile")
        os.makedirs(profile_dir, exist_ok=True)

        with sync_playwright() as p:
            launch_args = {
                "headless": self.headless,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-web-security",
                    "--disable-infobars",
                    "--window-size=1280,800",
                    "--disable-dev-shm-usage",
                ],
            }
            if self.proxy:
                launch_args["proxy"] = {"server": self.proxy}

            browser = p.chromium.launch(**launch_args)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="America/New_York",
                color_scheme="light",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )

            # 应用 stealth 补丁 (隐藏 webdriver/automation 特征)
            try:
                from playwright_stealth import stealth_sync
                stealth_sync(context)
                logger.info("[Browser] stealth 补丁已应用")
            except ImportError:
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """)

            # 额外的反检测注入
            context.add_init_script("""
                // 模拟真实 Chrome 的 plugin 列表
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin' },
                    ],
                });
                // 模拟 WebGL
                const getParameter = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) return 'Intel Inc.';
                    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
                    return getParameter.call(this, parameter);
                };
                // chrome.runtime 存在性
                if (!window.chrome) window.chrome = {};
                if (!window.chrome.runtime) window.chrome.runtime = { id: undefined };
            """)

            page = context.new_page()
            page.set_default_timeout(timeout * 1000 + 15000)

            try:
                # 在 Stripe hosted 域下挂载虚拟页面，确保 hCaptcha 校验上下文域名一致
                page.route(route_pattern, lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body=hcaptcha_html,
                ))

                page.goto(target_nav_url, wait_until="domcontentloaded", timeout=15000)

                # 模拟用户行为 — 鼠标移动、滚动 (提升 hCaptcha 评分)
                logger.info("[Browser] 模拟用户行为...")
                time.sleep(random.uniform(0.5, 1.5))

                # 随机鼠标移动 (模拟人类不规则轨迹)
                for _ in range(random.randint(3, 6)):
                    x = random.randint(100, 800)
                    y = random.randint(100, 600)
                    page.mouse.move(x, y, steps=random.randint(5, 15))
                    time.sleep(random.uniform(0.1, 0.4))

                # 模拟页面滚动
                page.evaluate("window.scrollBy(0, 100)")
                time.sleep(random.uniform(0.3, 0.8))
                page.evaluate("window.scrollBy(0, -50)")
                time.sleep(random.uniform(0.3, 0.6))

                # 模拟点击页面空白处
                page.mouse.click(random.randint(200, 400), random.randint(300, 500))
                time.sleep(random.uniform(0.5, 1.0))

                # 等待 hCaptcha SDK 加载
                logger.info("[Browser] 等待 hCaptcha SDK 加载...")
                page.wait_for_function("window.__hcaptchaReady === true", timeout=15000)
                logger.info("[Browser] hCaptcha SDK 已加载")

                # 更多鼠标移动 (在 hCaptcha 容器附近)
                for _ in range(random.randint(2, 4)):
                    x = random.randint(200, 500)
                    y = random.randint(350, 550)
                    page.mouse.move(x, y, steps=random.randint(8, 20))
                    time.sleep(random.uniform(0.1, 0.3))

                time.sleep(random.uniform(0.5, 1.5))

                # 渲染 + 执行 invisible hCaptcha
                logger.info("[Browser] 执行 hCaptcha invisible challenge...")
                timeout_ms = timeout * 1000
                result = page.evaluate("""
                    (params) => {
                        return new Promise((resolve, reject) => {
                            const timer = setTimeout(() => {
                                reject(new Error('hCaptcha execute timeout'));
                            }, params.timeout);

                            try {
                                const widgetId = hcaptcha.render('hcaptcha-container', {
                                    sitekey: params.siteKey,
                                    size: 'invisible',
                                    callback: (token) => {
                                        clearTimeout(timer);
                                        const ekey = hcaptcha.getRespKey(widgetId) || '';
                                        resolve({ success: true, token: token, ekey: ekey });
                                    },
                                    'expired-callback': () => {
                                        clearTimeout(timer);
                                        resolve({success: false, error: 'hCaptcha token expired'});
                                    },
                                    'chalexpired-callback': () => {
                                        clearTimeout(timer);
                                        resolve({success: false, error: 'hCaptcha challenge expired'});
                                    },
                                    'error-callback': (err) => {
                                        clearTimeout(timer);
                                        resolve({success: false, error: 'hCaptcha error: ' + String(err)});
                                    },
                                });

                                const executeOpts = {};
                                if (params.rqdata) {
                                    executeOpts.rqdata = params.rqdata;
                                }
                                hcaptcha.execute(widgetId, executeOpts);
                            } catch (e) {
                                clearTimeout(timer);
                                resolve({success: false, error: e.message || String(e)});
                            }
                        });
                    }
                """, {"siteKey": site_key, "rqdata": rqdata, "timeout": timeout_ms})

                logger.info(f"[Browser] hCaptcha 结果: success={result.get('success')}, token_len={len(result.get('token', ''))}")
                return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[Browser] hCaptcha 异常: {error_msg}")
                try:
                    page.screenshot(path="test_outputs/browser_hcaptcha_error.png")
                except Exception:
                    pass
                return {"success": False, "error": f"Browser exception: {error_msg}"}
            finally:
                browser.close()

    def solve_hcaptcha_uc(self, site_key: str, site_url: str, rqdata: str = "", timeout: int = 60) -> dict:
        """
        使用 undetected-chromedriver (Selenium) 执行 hCaptcha invisible 挑战。
        UC 专门为绕过 hCaptcha/CloudFlare 等 bot 检测设计。
        """
        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            return {"success": False, "error": "undetected-chromedriver not installed"}

        import os, random, tempfile

        logger.info(f"[UC] 启动 undetected Chrome: site_key={site_key[:20]}...")
        logger.info(f"[UC] headless={self.headless}, proxy={self.proxy or '直连'}")

        # 持久化 profile
        profile_dir = os.path.join(os.path.dirname(__file__), ".uc_profile")
        os.makedirs(profile_dir, exist_ok=True)

        # hCaptcha 页面
        hcaptcha_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Secure Verification</title>
    <script src="https://js.hcaptcha.com/1/api.js?render=explicit&onload=onHcaptchaLoad" async defer></script>
    <style>body { font-family: sans-serif; padding: 40px; background: #f6f9fc; }</style>
</head>
<body>
    <h2>Verifying your identity</h2>
    <div id="hcaptcha-container"></div>
    <script>
        window.__hcaptchaReady = false;
        function onHcaptchaLoad() { window.__hcaptchaReady = true; }
    </script>
</body>
</html>"""

        # 保存 HTML 到临时文件 (UC 不支持 route interception)
        html_file = os.path.join(tempfile.gettempdir(), "hcaptcha_challenge.html")
        with open(html_file, "w") as f:
            f.write(hcaptcha_html)

        options = uc.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-extensions")
        options.add_argument(f"--user-data-dir={profile_dir}")
        if self.proxy:
            options.add_argument(f"--proxy-server={self.proxy}")

        driver = None
        try:
            driver = uc.Chrome(
                options=options,
                use_subprocess=True,
            )
            driver.set_page_load_timeout(30)

            # 先访问 js.stripe.com 获得正确的域上下文
            logger.info("[UC] 导航到 js.stripe.com...")
            driver.get("https://js.stripe.com/v3/")
            time.sleep(random.uniform(1.0, 2.0))

            # 在 Stripe 域下注入 hCaptcha 页面
            logger.info("[UC] 注入 hCaptcha 环境...")
            driver.execute_script("""
                document.head.innerHTML = '';
                document.body.innerHTML = '<div id="hcaptcha-container"></div>';
                window.__hcaptchaReady = false;
                var script = document.createElement('script');
                script.src = 'https://js.hcaptcha.com/1/api.js?render=explicit&onload=onHcaptchaLoad';
                script.async = true;
                window.onHcaptchaLoad = function() { window.__hcaptchaReady = true; };
                document.head.appendChild(script);
            """)

            # 模拟用户行为
            time.sleep(random.uniform(1.0, 2.0))
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(driver)
            for _ in range(random.randint(3, 6)):
                x = random.randint(-300, 300)
                y = random.randint(-200, 200)
                actions.move_by_offset(x, y).perform()
                actions = ActionChains(driver)  # reset
                time.sleep(random.uniform(0.1, 0.3))

            # 等待 hCaptcha SDK 加载
            logger.info("[UC] 等待 hCaptcha SDK 加载...")
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return window.__hcaptchaReady === true")
            )
            logger.info("[UC] hCaptcha SDK 已加载")

            # 更多鼠标移动
            for _ in range(random.randint(2, 4)):
                actions = ActionChains(driver)
                actions.move_by_offset(random.randint(-100, 100), random.randint(-50, 50)).perform()
                time.sleep(random.uniform(0.1, 0.3))

            time.sleep(random.uniform(0.5, 1.0))

            # 执行 hCaptcha challenge
            logger.info("[UC] 执行 hCaptcha invisible challenge...")
            timeout_ms = timeout * 1000
            result = driver.execute_script("""
                return new Promise((resolve, reject) => {
                    const timer = setTimeout(() => {
                        resolve({success: false, error: 'hCaptcha execute timeout'});
                    }, arguments[0].timeout);

                    try {
                        const widgetId = hcaptcha.render('hcaptcha-container', {
                            sitekey: arguments[0].siteKey,
                            size: 'invisible',
                            callback: (token) => {
                                clearTimeout(timer);
                                const ekey = hcaptcha.getRespKey(widgetId) || '';
                                resolve({ success: true, token: token, ekey: ekey });
                            },
                            'error-callback': (err) => {
                                clearTimeout(timer);
                                resolve({success: false, error: 'hCaptcha error: ' + String(err)});
                            },
                        });

                        const executeOpts = {};
                        if (arguments[0].rqdata) {
                            executeOpts.rqdata = arguments[0].rqdata;
                        }
                        hcaptcha.execute(widgetId, executeOpts);
                    } catch (e) {
                        clearTimeout(timer);
                        resolve({success: false, error: e.message || String(e)});
                    }
                });
            """, {"siteKey": site_key, "rqdata": rqdata, "timeout": timeout_ms})

            logger.info(f"[UC] hCaptcha 结果: success={result.get('success')}, token_len={len(result.get('token', ''))}")
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[UC] 异常: {error_msg}")
            if driver:
                try:
                    driver.save_screenshot("test_outputs/uc_error.png")
                except Exception:
                    pass
            return {"success": False, "error": f"UC exception: {error_msg}"}
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def solve(self, pi_client_secret: str, timeout: int = 60, challenge_url: str = "") -> dict:
        """
        使用 Playwright 浏览器处理 Stripe hCaptcha 挑战。

        参数:
            pi_client_secret: payment_intent 的 client_secret (pi_xxx_secret_yyy)
            timeout: 最长等待时间 (秒)

        返回:
            {"success": True, "status": "succeeded|processing", "pi_data": {...}}
            {"success": False, "error": "..."}
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {"success": False, "error": "playwright 未安装: pip install playwright && playwright install chromium"}
        import os

        logger.info(f"[Browser] 启动 Chromium 处理 hCaptcha 挑战...")
        logger.info(f"[Browser] headless={self.headless}, proxy={self.proxy or '直连'}, timeout={timeout}s")

        # 构建要注入浏览器的HTML (加载 Stripe.js + 执行 handleNextAction)
        stripe_html = self._build_stripe_page()

        with sync_playwright() as p:
            launch_args = {
                "headless": self.headless,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-web-security",  # 允许跨域 (虚拟HTTPS页面加载Stripe.js)
                    "--disable-infobars",
                    "--window-size=1280,800",
                ],
            }
            if self.proxy:
                launch_args["proxy"] = {"server": self.proxy}

            browser = p.chromium.launch(**launch_args)
            context = browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )

            # 应用 stealth 补丁 (隐藏 webdriver/automation 特征)
            try:
                from playwright_stealth import stealth_sync
                stealth_sync(context)
                logger.info("[Browser] stealth 补丁已应用")
            except ImportError:
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """)

            page = context.new_page()
            # 全局超时设置
            page.set_default_timeout(timeout * 1000 + 15000)  # JS timeout + 额外缓冲

            try:
                # 每轮 solve 重置 challenge 状态跟踪。
                self._last_checkcaptcha_pass = None
                self._verify_blocked_count = 0
                self._last_generated_pass_uuid = ""
                self._last_checkcaptcha_ekey = ""
                self._last_verify_request_body = ""
                self._last_getcaptcha_ts = 0.0

                # 用 route 拦截创建虚拟 HTTPS 页面
                # 优先使用 checkout host（更接近真实上下文），fallback 到 js.stripe.com
                target_url = (challenge_url or "").strip() or "https://checkout.stripe.com/challenge/checkout"
                parsed = urlsplit(target_url)
                if not parsed.scheme or not parsed.netloc:
                    target_url = "https://checkout.stripe.com/challenge/checkout"
                    parsed = urlsplit(target_url)
                target_nav_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
                route_pattern = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}*"
                use_real_page = os.getenv("BROWSER_CHALLENGE_USE_REAL_PAGE", "1") not in ("0", "false", "False")
                if use_real_page:
                    logger.info("[Browser] 使用真实 Stripe 页面上下文: %s", target_nav_url[:140])
                else:
                    logger.info("[Browser] 创建虚拟 HTTPS 页面: %s", target_nav_url[:140])
                    page.route(route_pattern, lambda route: route.fulfill(
                        status=200,
                        content_type="text/html",
                        body=stripe_html,
                    ))

                # 监听浏览器 console 输出
                page.on("console", lambda msg: logger.info(f"[Browser console] {msg.type}: {msg.text}"))
                page.on("pageerror", lambda err: logger.error(f"[Browser error] {err}"))
                # 监听网络请求: 特别关注 verify_challenge 请求
                def _log_response(resp):
                    url = resp.url
                    if resp.status >= 400:
                        logger.warning(f"[Browser {resp.status}] {url[:150]}")
                    max_log_len = 1800
                    try:
                        max_log_len = max(400, min(12000, int(os.getenv("HCAPTCHA_API_LOG_MAXLEN", "1800"))))
                    except Exception:
                        max_log_len = 1800

                    json_body = ""
                    if "hcaptcha.com" in url:
                        try:
                            ctype = (resp.headers.get("content-type", "") or "").lower()
                        except Exception:
                            ctype = ""
                        # 仅记录 JSON 响应，避免日志淹没
                        if "json" in ctype:
                            try:
                                body = resp.text()
                                json_body = body
                                logger.info(f"[Browser] hCaptcha API 响应: {url[:160]} body={body[:max_log_len]}")
                            except Exception:
                                pass
                    if "hcaptcha.com/getcaptcha" in url or "/checkcaptcha/" in url:
                        try:
                            body = json_body or resp.text()
                            logger.info(f"[Browser] hCaptcha challenge 响应(强制记录): {url[:160]} body={body[:max_log_len]}")
                            if "/checkcaptcha/" in url:
                                try:
                                    pz = urlsplit(url)
                                    segs = [s for s in (pz.path or "").split("/") if s]
                                    # /checkcaptcha/<sitekey>/<ekey>
                                    if len(segs) >= 3:
                                        self._last_checkcaptcha_ekey = segs[-1]
                                except Exception:
                                    pass
                                try:
                                    cp = json.loads(body)
                                except Exception:
                                    cp = {}
                                if isinstance(cp, dict):
                                    if isinstance(cp.get("pass"), bool):
                                        self._last_checkcaptcha_pass = bool(cp.get("pass"))
                                        logger.info("[Browser] checkcaptcha pass=%s", self._last_checkcaptcha_pass)
                                        if self._last_checkcaptcha_pass:
                                            gp = (cp.get("generated_pass_UUID") or "").strip()
                                            if gp:
                                                self._last_generated_pass_uuid = gp
                                                logger.info(
                                                    "[Browser] checkcaptcha generated_pass_UUID 已缓存 len=%s ekey=%s",
                                                    len(gp),
                                                    (self._last_checkcaptcha_ekey or "")[:24],
                                                )
                                    elif cp.get("success") is False:
                                        self._last_checkcaptcha_pass = False
                                        logger.info("[Browser] checkcaptcha success=false")
                            if "hcaptcha.com/getcaptcha" in url:
                                try:
                                    parsed = json.loads(body)
                                except Exception:
                                    parsed = {}
                                if isinstance(parsed, dict) and parsed:
                                    self._last_getcaptcha_payload = parsed
                                    self._last_getcaptcha_ts = time.time()
                                    net_prompt = ""
                                    q = parsed.get("requester_question")
                                    if isinstance(q, dict):
                                        net_prompt = (
                                            q.get("en")
                                            or q.get("default")
                                            or next((str(v).strip() for v in q.values() if str(v).strip()), "")
                                        )
                                    elif isinstance(q, str):
                                        net_prompt = q.strip()

                                    if not net_prompt:
                                        for key in ("prompt", "question", "request_type"):
                                            v = parsed.get(key)
                                            if isinstance(v, str) and v.strip():
                                                net_prompt = v.strip()
                                                break

                                    if net_prompt:
                                        self._last_net_prompt = net_prompt
                                        logger.info("[Browser] getcaptcha 题面(网络): %s", net_prompt[:240])

                                # 关键调试：保留完整 getcaptcha body，便于分析 task schema/prompt。
                                try:
                                    os.makedirs("test_outputs", exist_ok=True)
                                    ts = int(time.time() * 1000)
                                    out_path = f"test_outputs/hcaptcha_getcaptcha_{ts}.json"
                                    with open(out_path, "w", encoding="utf-8") as f:
                                        f.write(body)
                                    logger.info("[Browser] 已保存 getcaptcha 响应: %s", out_path)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    if "verify_challenge" in url:
                        logger.info(f"[Browser] verify_challenge 请求: {url}")
                        logger.info(f"[Browser] verify_challenge 状态: {resp.status}")
                        try:
                            body = resp.text()
                            logger.info(f"[Browser] verify_challenge 响应: {body[:500]}")
                            try:
                                os.makedirs("test_outputs", exist_ok=True)
                                ts = int(time.time() * 1000)
                                out_path = f"test_outputs/verify_challenge_resp_{ts}.json"
                                with open(out_path, "w", encoding="utf-8") as f:
                                    f.write(body)
                                logger.info("[Browser] 已保存 verify_challenge 响应: %s", out_path)
                            except Exception:
                                pass
                        except Exception:
                            pass
                def _log_request(req):
                    if "verify_challenge" in req.url:
                        logger.info(f"[Browser] verify_challenge POST: {req.url}")
                        logger.info(f"[Browser] verify_challenge headers: {dict(req.headers)}")
                        if req.post_data:
                            # 打印完整 body (解析字段名)
                            body = req.post_data
                            self._last_verify_request_body = body
                            logger.info(f"[Browser] verify_challenge body (len={len(body)}): {body[:200]}")
                            # 解析并列出所有字段名
                            fields = [part.split('=')[0] for part in body.split('&') if '=' in part]
                            logger.info(f"[Browser] verify_challenge 字段: {fields}")
                            try:
                                kv = dict(parse_qsl(body, keep_blank_values=True))
                            except Exception:
                                kv = {}
                            if kv:
                                tok = (kv.get("challenge_response_token") or "").strip()
                                ekey = (kv.get("challenge_response_ekey") or "").strip()
                                logger.info(
                                    "[Browser] verify_challenge token_len=%s ekey=%s vendor=%s",
                                    len(tok),
                                    ekey[:32],
                                    (kv.get("captcha_vendor_name") or "")[:16],
                                )
                            try:
                                os.makedirs("test_outputs", exist_ok=True)
                                ts = int(time.time() * 1000)
                                out_path = f"test_outputs/verify_challenge_req_{ts}.txt"
                                with open(out_path, "w", encoding="utf-8") as f:
                                    f.write(body)
                                logger.info("[Browser] 已保存 verify_challenge 请求体: %s", out_path)
                            except Exception:
                                pass
                    if "api.hcaptcha.com/checkcaptcha/" in req.url:
                        body = req.post_data or ""
                        if body:
                            max_req_log = 1500
                            try:
                                max_req_log = max(300, min(12000, int(os.getenv("HCAPTCHA_API_REQ_LOG_MAXLEN", "1500"))))
                            except Exception:
                                max_req_log = 1500
                            logger.info(f"[Browser] checkcaptcha POST: {req.url[:180]}")
                            logger.info(f"[Browser] checkcaptcha body (len={len(body)}): {body[:max_req_log]}")
                            try:
                                cp_req = json.loads(body)
                            except Exception:
                                cp_req = {}
                            if isinstance(cp_req, dict):
                                jm = str(cp_req.get("job_mode") or "").strip().lower()
                                if jm:
                                    # 在 getcaptcha 丢失/延迟时，用最近一次 checkcaptcha 的 mode 做保底。
                                    self._last_getcaptcha_payload = {
                                        **(self._last_getcaptcha_payload if isinstance(self._last_getcaptcha_payload, dict) else {}),
                                        "request_type": jm,
                                    }
                                    self._last_getcaptcha_ts = time.time()
                            try:
                                os.makedirs("test_outputs", exist_ok=True)
                                ts = int(time.time() * 1000)
                                out_path = f"test_outputs/hcaptcha_checkcaptcha_req_{ts}.txt"
                                with open(out_path, "w", encoding="utf-8") as f:
                                    f.write(body)
                                logger.info("[Browser] 已保存 checkcaptcha 请求体: %s", out_path)
                            except Exception:
                                pass
                def _route_verify(route, request):
                    try:
                        mode = (os.getenv("BROWSER_CHALLENGE_BLOCK_VERIFY", "") or "").strip().lower()
                        if mode in ("", "0", "false", "off", "none"):
                            route.continue_()
                            return

                        if mode in ("always", "1", "true", "force"):
                            should_block = True
                        elif mode in ("until_pass", "wait_pass", "gate"):
                            should_block = bool(self._last_checkcaptcha_pass is not True)
                        else:
                            should_block = False

                        if should_block:
                            self._verify_blocked_count = int(self._verify_blocked_count or 0) + 1
                            logger.info(
                                "[Browser] 拦截 verify_challenge mode=%s checkcaptcha_pass=%s blocked=%s",
                                mode,
                                self._last_checkcaptcha_pass,
                                self._verify_blocked_count,
                            )
                            route.abort()
                            return
                        route.continue_()
                    except Exception:
                        try:
                            route.continue_()
                        except Exception:
                            pass
                page.on("response", _log_response)
                page.on("request", _log_request)
                page.route("**/verify_challenge*", _route_verify)

                page.goto(target_nav_url, wait_until="domcontentloaded", timeout=30000)

                # 真实页面上下文下，若 Stripe 未暴露则兜底注入 v3。
                has_stripe = False
                try:
                    has_stripe = bool(page.evaluate("typeof Stripe !== 'undefined'"))
                except Exception:
                    has_stripe = False
                if not has_stripe:
                    try:
                        page.add_script_tag(url="https://js.stripe.com/v3/")
                    except Exception:
                        pass

                # 等待 Stripe.js v3 加载
                logger.info("[Browser] 等待 Stripe.js 加载...")
                page.wait_for_function("typeof Stripe !== 'undefined'", timeout=30000)
                logger.info("[Browser] Stripe.js 已加载")

                # 执行 handleNextAction（异步）并在等待期间持续尝试点击 hCaptcha
                # 目的：避免 Promise 一直 pending 直到超时（通常是 challenge UI 未被触发交互）。
                logger.info("[Browser] 调用 stripe.handleNextAction()...")
                timeout_ms = timeout * 1000

                page.evaluate("""
                    (params) => {
                        window.__hnaDone = false;
                        window.__hnaResult = null;
                        (async () => {
                            const stripe = Stripe(params.pk);
                            try {
                                const timeoutPromise = new Promise((_, reject) =>
                                    setTimeout(() => reject(new Error('handleNextAction timeout')), params.timeout)
                                );
                                const actionPromise = stripe.handleNextAction({
                                    clientSecret: params.clientSecret,
                                });
                                const result = await Promise.race([actionPromise, timeoutPromise]);

                                if (result && result.error) {
                                    window.__hnaResult = {
                                        success: false,
                                        error: result.error.message || result.error.type || 'handleNextAction_error',
                                        error_code: result.error.code || '',
                                    };
                                } else {
                                    const pi = (result && (result.paymentIntent || result.setupIntent)) || {};
                                    window.__hnaResult = {
                                        success: pi.status === 'succeeded' || pi.status === 'processing',
                                        status: pi.status || '',
                                        pi_id: pi.id || '',
                                        pi_data: {
                                            id: pi.id || '',
                                            status: pi.status || '',
                                            amount: pi.amount || null,
                                            currency: pi.currency || '',
                                        },
                                    };
                                }
                            } catch (e) {
                                window.__hnaResult = {
                                    success: false,
                                    error: e.message || String(e),
                                };
                            } finally {
                                window.__hnaDone = true;
                            }
                        })();
                    }
                """, {"pk": self.stripe_pk, "clientSecret": pi_client_secret, "timeout": timeout_ms})

                started = time.time()
                click_count = 0
                last_click_ts = 0.0
                solved_logged = False
                captured_token = ""
                token_source = ""
                try:
                    click_interval = max(2.0, float(os.getenv("HCAPTCHA_CLICK_INTERVAL", "8")))
                except Exception:
                    click_interval = 8.0
                try:
                    max_clicks = max(1, int(os.getenv("HCAPTCHA_MAX_CLICKS", "10")))
                except Exception:
                    max_clicks = 10
                while (time.time() - started) < (timeout + 10):
                    try:
                        done = page.evaluate("window.__hnaDone === true")
                    except Exception:
                        done = False
                    if done:
                        break

                    # 可选：一旦拿到 checkcaptcha pass token 就提前返回上层手动 verify，
                    # 用于排查 Stripe.js 内部 verify 失败问题。
                    try:
                        return_on_pass = os.getenv("BROWSER_CHALLENGE_RETURN_ON_PASS", "0") not in ("0", "false", "False")
                    except Exception:
                        return_on_pass = False
                    if return_on_pass and self._last_checkcaptcha_pass is True and self._last_generated_pass_uuid:
                        logger.info(
                            "[Browser] 命中 RETURN_ON_PASS: token_len=%s ekey=%s",
                            len(self._last_generated_pass_uuid),
                            (self._last_checkcaptcha_ekey or "")[:24],
                        )
                        return {
                            "success": False,
                            "error": "hcaptcha_pass_captured",
                            "hcaptcha_token": self._last_generated_pass_uuid,
                            "hcaptcha_ekey": self._last_checkcaptcha_ekey,
                            "hcaptcha_token_source": "checkcaptcha.generated_pass_UUID",
                        }

                    # 尝试从 hCaptcha frame 回捞 response token。
                    if not captured_token:
                        try:
                            tok = self._extract_hcaptcha_response(page)
                        except Exception:
                            tok = {}
                        if tok and tok.get("token"):
                            captured_token = tok.get("token", "")
                            token_source = tok.get("source", "")
                            logger.info(
                                "[Browser] 已捕获 hCaptcha token(len=%s) source=%s",
                                len(captured_token),
                                token_source[:90],
                            )

                    # 若 checkbox 已处于通过态，避免继续点击导致 challenge 重置。
                    try:
                        solved = self._is_hcaptcha_checked(page)
                    except Exception:
                        solved = False
                    if solved and not solved_logged:
                        solved_logged = True
                        logger.info("[Browser] 检测到 hCaptcha checkbox 已勾选，暂停主动点击并等待 handleNextAction 回调")

                    # 点击节流（保守）:
                    # - 默认每次点击后等待 8s（可由 HCAPTCHA_CLICK_INTERVAL 调整）
                    # - 默认最多 10 次（可由 HCAPTCHA_MAX_CLICKS 调整）
                    now = time.time()
                    if (not solved) and click_count < max_clicks and (now - last_click_ts) >= click_interval:
                        try:
                            clicked = self._try_click_hcaptcha(page)
                            if clicked:
                                click_count += 1
                                last_click_ts = now
                                logger.info(
                                    "[Browser] hCaptcha 交互点击尝试: %s/%s (interval=%ss)",
                                    click_count,
                                    max_clicks,
                                    int(click_interval),
                                )
                        except Exception as click_err:
                            logger.debug("[Browser] hCaptcha 点击异常: %s", click_err)

                    time.sleep(0.5)

                try:
                    result = page.evaluate("window.__hnaResult")
                except Exception:
                    result = None
                if not result:
                    result = {"success": False, "error": "handleNextAction timeout"}

                # 若 handleNextAction 未回调，但已拿到 hCaptcha token，带回上层尝试 verify_challenge。
                if (not result.get("success")) and captured_token:
                    result["hcaptcha_token"] = captured_token
                    result["hcaptcha_token_source"] = token_source

                # 回退：若页面 token 未抓到，但网络层已有 checkcaptcha 通过 token，同样回传。
                if (not result.get("success")) and (not result.get("hcaptcha_token")) and self._last_generated_pass_uuid:
                    result["hcaptcha_token"] = self._last_generated_pass_uuid
                    result["hcaptcha_ekey"] = self._last_checkcaptcha_ekey
                    result["hcaptcha_token_source"] = "checkcaptcha.generated_pass_UUID"

                logger.info(f"[Browser] handleNextAction 结果: {json.dumps(result, ensure_ascii=False)[:500]}")
                return result

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[Browser] Playwright 异常: {error_msg}")
                # 尝试截图保存调试信息
                try:
                    page.screenshot(path="test_outputs/browser_challenge_error.png")
                    logger.info("[Browser] 错误截图已保存: test_outputs/browser_challenge_error.png")
                except Exception:
                    pass
                return {"success": False, "error": f"Browser exception: {error_msg}"}
            finally:
                browser.close()

    def _extract_hcaptcha_response(self, page) -> dict:
        """
        从已加载的 hCaptcha frame 中提取 response token（若存在）。
        """
        for frame in page.frames:
            url = (frame.url or "").lower()
            if "hcaptcha" not in url:
                continue
            try:
                data = frame.evaluate(
                    """
                    () => {
                      const ta =
                        document.querySelector("textarea[name='h-captcha-response']") ||
                        document.querySelector("textarea[name='g-recaptcha-response']") ||
                        document.querySelector("textarea[id*='h-captcha-response']") ||
                        document.querySelector("textarea[id*='g-recaptcha-response']");
                      const token = (ta && ta.value ? ta.value.trim() : "");
                      return { token };
                    }
                    """
                ) or {}
                token = (data.get("token") or "").strip()
                if len(token) > 40:
                    return {"token": token, "source": frame.url or ""}
            except Exception:
                continue
        return {}

    def _build_stripe_page(self) -> str:
        """构建注入浏览器的 HTML 页面"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Payment Processing</title>
    <script src="https://js.stripe.com/v3/"></script>
</head>
<body>
    <div id="status">Loading Stripe.js...</div>
    <div id="hcaptcha-container"></div>
    <script>
        document.getElementById('status').textContent = 'Stripe.js loaded, ready.';
    </script>
</body>
</html>"""

    def _is_hcaptcha_checked(self, page) -> bool:
        """
        粗略检测 hCaptcha 是否已进入“已勾选/已通过”状态。
        """
        h_frames = [f for f in page.frames if "hcaptcha" in (f.url or "").lower()]
        if not h_frames:
            return False

        for frame in h_frames:
            try:
                checked = frame.evaluate(
                    """
                    () => {
                      const sels = [
                        "#checkbox[aria-checked='true']",
                        "[role='checkbox'][aria-checked='true']",
                        "[aria-checked='true']",
                      ];
                      for (const s of sels) {
                        if (document.querySelector(s)) return true;
                      }
                      return false;
                    }
                    """
                )
                if checked:
                    return True
            except Exception:
                continue
        return False

    def _try_click_hcaptcha(self, page) -> bool:
        """
        检测并尝试点击 hCaptcha 相关 frame 中的 checkbox/challenge 区域。
        与 browser_payment 中策略保持一致，优先 newassets.hcaptcha.com checkbox frame。
        """
        h_frames = [f for f in page.frames if "hcaptcha" in (f.url or "").lower()]
        if not h_frames:
            return False

        def _prio(u: str) -> int:
            u = (u or "").lower()
            if "newassets.hcaptcha.com" in u and "frame=checkbox" in u:
                return 0
            if "newassets.hcaptcha.com" in u and "frame=challenge" in u:
                return 1
            if "newassets.hcaptcha.com" in u:
                return 2
            if "b.stripecdn.com" in u:
                return 3
            if "js.stripe.com" in u:
                return 4
            return 5

        h_frames = sorted(h_frames, key=lambda f: _prio(f.url or ""))
        challenge_frames = [f for f in h_frames if "frame=challenge" in (f.url or "").lower()]
        any_click = False

        # 关键策略：
        # 一旦 challenge frame 已出现，暂停对 checkbox 的持续点击，
        # 避免在 challenge 进行中反复重置状态。
        if challenge_frames:
            for frame in challenge_frames:
                url = frame.url or ""
                # 仅处理“可见且可交互”的 challenge iframe。
                # hCaptcha 常会保留历史 challenge frame（隐藏状态），
                # 若误操作这些旧 frame，会出现题面卡死/循环提问。
                try:
                    frame_el = frame.frame_element()
                except Exception:
                    frame_el = None
                if frame_el is None:
                    continue
                try:
                    box = frame_el.bounding_box() or {}
                    if (
                        float(box.get("width") or 0) < 140
                        or float(box.get("height") or 0) < 140
                    ):
                        continue
                    frame_visible = frame_el.evaluate(
                        """
                        (el) => {
                          const r = el.getBoundingClientRect();
                          const cs = getComputedStyle(el);
                          const op = parseFloat(cs.opacity || "1");
                          if (!isFinite(op)) return false;
                          if (cs.display === "none" || cs.visibility === "hidden" || op < 0.05) return false;
                          if (r.width < 140 || r.height < 140) return false;
                          if (r.bottom < 0 || r.right < 0) return false;
                          return true;
                        }
                        """
                    )
                    if not frame_visible:
                        continue
                except Exception:
                    # 可见性判断异常时，保守继续旧逻辑。
                    pass

                try:
                    meta = frame.evaluate(
                        """
                        () => ({
                          w: Math.max(0, window.innerWidth || 0),
                          h: Math.max(0, window.innerHeight || 0),
                          t: ((document.body && document.body.innerText) || "").slice(0, 160),
                        })
                        """
                    ) or {}
                except Exception:
                    meta = {}
                c_w = int(meta.get("w") or 0)
                c_h = int(meta.get("h") or 0)
                # 预加载/隐藏 challenge frame（常见为 0x0）不做中心点击，
                # 让流程回退到 checkbox 触发真正 challenge。
                if c_w < 200 or c_h < 200:
                    continue
                try:
                    attempted = self._attempt_visual_challenge(page, frame, url)
                    if attempted:
                        any_click = True
                        continue
                except Exception as e:
                    logger.debug("[Browser] visual challenge attempt 异常: %s", e)

                # challenge 已出现但尚未进入可点击态时，做一次轻量中心点击唤醒。
                try:
                    # 对于已完整渲染的大 challenge（>420x420），避免频繁中心点击干扰题面状态。
                    if c_w >= 420 and c_h >= 420:
                        continue
                    clicked_center = frame.evaluate(
                        """
                        () => {
                          const x = Math.floor(window.innerWidth / 2);
                          const y = Math.floor(window.innerHeight / 2);
                          const el = document.elementFromPoint(x, y) || document.body;
                          if (!el) return false;
                          const opts = {bubbles: true, cancelable: true, view: window, clientX: x, clientY: y};
                          el.dispatchEvent(new MouseEvent('mousemove', opts));
                          el.dispatchEvent(new MouseEvent('mousedown', opts));
                          el.dispatchEvent(new MouseEvent('mouseup', opts));
                          el.dispatchEvent(new MouseEvent('click', opts));
                          return true;
                        }
                        """
                    )
                    if clicked_center:
                        logger.info(f"[Browser] hCaptcha 已尝试点击 challenge frame 中心: {url[:90]}")
                        any_click = True
                except Exception:
                    pass

            # 只有在 challenge frame 确实发生交互时才提前返回。
            # 否则可能只是隐藏预加载 frame，仍需回退点击 checkbox 触发真正 challenge。
            if any_click:
                return True

        # 没有 challenge frame 时，才点击 checkbox。
        for frame in h_frames:
            url = frame.url or ""
            try:
                # 已勾选状态下不再重复点击，避免重置 challenge。
                already_checked = frame.evaluate(
                    """
                    () => {
                      const el =
                        document.querySelector("#checkbox[aria-checked='true']") ||
                        document.querySelector("[role='checkbox'][aria-checked='true']");
                      return !!el;
                    }
                    """
                )
                if already_checked:
                    continue

                for sel in [
                    "#checkbox",
                    "[id*='checkbox']",
                    "[role='checkbox']",
                    "#anchor",
                    ".check",
                    ".checkbox",
                    "[aria-checked]",
                ]:
                    el = frame.query_selector(sel)
                    if el:
                        # 避免短时间内重复点击 checkbox 导致 challenge 状态重置。
                        if (time.time() - float(self._last_checkbox_click_ts or 0.0)) < 8.0:
                            break
                        el.click(force=True, timeout=800)
                        self._last_checkbox_click_ts = time.time()
                        logger.info(f"[Browser] hCaptcha 点击成功: {sel} @ {url[:90]}")
                        any_click = True
                        break
            except Exception:
                continue

        return any_click

    def _manual_visual_step(
        self,
        page,
        frame,
        frame_url: str,
        prompt: str,
        width: int,
        height: int,
        grid_rect: dict,
        submit_text: str = "",
        task_count: int = 0,
        body_text: str = "",
    ) -> bool:
        """
        人工介入模式（HCAPTCHA_MANUAL=1）:
        - 保存截图到 test_outputs/manual_hcaptcha_*.png
        - 终端提示输入九宫格编号（1-9）
        - 自动点击 Verify/Next/Skip 提交
        """
        import os

        # 优先从 DOM 抽取任务格中心（兼容 3x3 / 4x4 / 其它布局）
        dynamic_tiles = []
        selector_debug = {}
        task_assets = {"examples": [], "tasks": []}
        try:
            dyn = frame.evaluate(
                """
                () => {
                  const selectorCandidates = [
                    '.task, .task-image',
                    "[class*='task']",
                    "[class*='image']",
                    "[aria-label]",
                  ];

                  function collect(selector) {
                    const arr = [];
                    for (const el of Array.from(document.querySelectorAll(selector))) {
                      const r = el.getBoundingClientRect();
                      const cs = getComputedStyle(el);
                      if (r.width < 22 || r.height < 22) continue;
                      if (r.width > 220 || r.height > 220) continue;
                      if (cs.display === 'none' || cs.visibility === 'hidden') continue;
                      arr.push({
                        x: r.left + r.width / 2,
                        y: r.top + r.height / 2,
                        area: r.width * r.height,
                      });
                    }
                    return arr;
                  }

                  let best = [];
                  const counts = {};
                  for (const sel of selectorCandidates) {
                    const cur = collect(sel);
                    counts[sel] = cur.length;
                    if (cur.length > best.length && cur.length <= 30) {
                      best = cur;
                    }
                  }
                  if (!best.length) return { points: [], counts };

                  // 去重（同一点附近多个元素重叠）
                  const uniq = [];
                  for (const p of best) {
                    const dup = uniq.find(u => Math.abs(u.x - p.x) < 6 && Math.abs(u.y - p.y) < 6);
                    if (!dup) uniq.push(p);
                  }

                  uniq.sort((a, b) => {
                    if (Math.abs(a.y - b.y) > 10) return a.y - b.y;
                    return a.x - b.x;
                  });
                  return {
                    points: uniq.map((p) => ({x: Math.round(p.x), y: Math.round(p.y)})),
                    counts,
                  };
                }
                """
            ) or {}
            points = dyn.get("points") if isinstance(dyn, dict) else dyn
            selector_debug = dyn.get("counts", {}) if isinstance(dyn, dict) else {}
            for p in (points or []):
                x = int(p.get("x") or 0)
                y = int(p.get("y") or 0)
                if x > 0 and y > 0:
                    dynamic_tiles.append((x, y))
        except Exception:
            dynamic_tiles = []

        # 抽取 challenge 资产 URL（若可见），便于后续做 URL 级识别。
        try:
            asset_info = frame.evaluate(
                """
                () => {
                  const ex = [];
                  const tasks = [];
                  const seenEx = new Set();
                  const seenTask = new Set();

                  const pullBgUrl = (el) => {
                    const bg = (getComputedStyle(el).backgroundImage || '') + ' ' + (el.style.backgroundImage || '');
                    const m = bg.match(/url\\((['"]?)(.*?)\\1\\)/i);
                    return m && m[2] ? m[2] : '';
                  };

                  const pushUrl = (arr, seen, url, extra={}) => {
                    if (!url || typeof url !== 'string') return;
                    if (!/^https?:\\/\\//i.test(url)) return;
                    if (seen.has(url)) return;
                    seen.add(url);
                    arr.push(Object.assign({url}, extra));
                  };

                  // 示例图（题目上方）
                  const exNodes = Array.from(document.querySelectorAll('[class*=\"example\"], [class*=\"prompt\"] img, [class*=\"prompt\"] [style*=\"background-image\"]'));
                  for (const el of exNodes) {
                    const src = el.getAttribute && (el.getAttribute('src') || el.getAttribute('data-src') || '');
                    const bg = pullBgUrl(el);
                    const url = src || bg;
                    pushUrl(ex, seenEx, url);
                  }

                  // 任务图
                  const taskNodes = Array.from(document.querySelectorAll('.task-image, .task, [class*=\"task\"]'));
                  for (const el of taskNodes) {
                    const src = el.getAttribute && (el.getAttribute('src') || el.getAttribute('data-src') || '');
                    const bg = pullBgUrl(el);
                    const url = src || bg;
                    const key =
                      (el.getAttribute && (el.getAttribute('data-task-key') || el.getAttribute('taskkey') || '')) ||
                      (el.dataset && (el.dataset.taskKey || '')) ||
                      '';
                    pushUrl(tasks, seenTask, url, {key});
                  }

                  return {examples: ex.slice(0, 8), tasks: tasks.slice(0, 40)};
                }
                """
            ) or {}
            if isinstance(asset_info, dict):
                ex = asset_info.get("examples") or []
                ts = asset_info.get("tasks") or []
                if isinstance(ex, list):
                    task_assets["examples"] = ex
                if isinstance(ts, list):
                    task_assets["tasks"] = ts
        except Exception:
            task_assets = {"examples": [], "tasks": []}

        # 提前拿到 frame 位置，便于后续从整页截图反推 challenge 对象中心。
        try:
            frame_el = frame.frame_element()
            box = frame_el.bounding_box() if frame_el else None
        except Exception:
            box = None

        # 估算网格维度:
        # - 动态点位可信时优先用动态数量推断 N
        # - 否则结合 prompt/task_count 兜底 (3x3 / 4x4)
        prompt_lc = (prompt or "").lower()
        body_text_lc = (body_text or "").lower()
        composite = f"{prompt_lc}\n{body_text_lc}"

        # 结合最近一轮 getcaptcha payload 做模式判定（但 payload 过旧时必须降权，避免串题）。
        try:
            payload_ttl = max(5.0, float(os.getenv("HCAPTCHA_GETCAPTCHA_TTL", "20")))
        except Exception:
            payload_ttl = 20.0
        last_payload_raw = self._last_getcaptcha_payload if isinstance(self._last_getcaptcha_payload, dict) else {}
        payload_age = (time.time() - float(self._last_getcaptcha_ts or 0.0)) if self._last_getcaptcha_ts else 1e9
        payload_fresh = bool(last_payload_raw and payload_age <= payload_ttl)
        last_payload = last_payload_raw if payload_fresh else {}
        raw_request_type = str(
            last_payload.get("request_type")
            or last_payload.get("job_mode")
            or ""
        ).strip().lower()
        net_request_type = raw_request_type
        req_cfg = last_payload.get("request_config") if isinstance(last_payload.get("request_config"), dict) else {}
        tasklist = last_payload.get("tasklist") if isinstance(last_payload.get("tasklist"), list) else []
        net_task_count = len(tasklist)
        shape_type = str(req_cfg.get("shape_type") or "").strip().lower()
        # 基础模式：仅在 payload 新鲜时优先信任网络 request_type。
        if payload_fresh and net_request_type in ("image_label_area_select", "image_label_binary", "image_drag_drop"):
            is_area_select_mode = net_request_type == "image_label_area_select"
            is_binary_mode = net_request_type == "image_label_binary"
        else:
            is_area_select_mode = (shape_type == "point")
            is_binary_mode = False
        area_prompt_hint = any(
            k in composite
            for k in (
                "point where the line breaks",
                "line breaks",
                "matching silhouette",
                "character in the middle",
                "click on the point",
                "identify the one that you can see in the example image",
            )
        )
        binary_prompt_hint = any(
            k in composite
            for k in (
                "tap on all",
                "choose everything",
                "identify everything",
                "you can see in the sample image",
                "you can see in the example image",
                "same place as the one shown",
                "could lift by hand",
            )
        )
        if (not payload_fresh) and area_prompt_hint:
            is_area_select_mode = True

        is_drag_challenge = any(
            k in composite
            for k in (
                "drag the segment",
                "drag segment",
                "complete the line",
                "drag the piece",
                "drag the object",
                "matching shadow",
                "slide the piece",
            )
        )
        # 网络层明确是拖拽题时，强制进入拖拽路径。
        if net_request_type == "image_drag_drop":
            is_drag_challenge = True

        # 模式兜底修正：
        # 1) 拖拽题优先级最高；
        # 2) 仅在 payload 不新鲜时，才用文案把题型兜底到 binary。
        if is_drag_challenge:
            is_area_select_mode = False
            is_binary_mode = False
        elif (not payload_fresh) and binary_prompt_hint and not area_prompt_hint:
            is_binary_mode = True
            is_area_select_mode = False

        logger.info(
            "[Browser][Manual] mode infer: net=%s fresh=%s age=%.1fs area=%s binary=%s drag=%s",
            net_request_type or "",
            payload_fresh,
            payload_age if payload_age < 999999 else -1.0,
            is_area_select_mode,
            is_binary_mode,
            is_drag_challenge,
        )

        dynamic_count = len(dynamic_tiles)
        dynamic_n = int(round(math.sqrt(dynamic_count))) if dynamic_count > 0 else 0
        dynamic_square_ok = (
            dynamic_n >= 2
            and dynamic_n * dynamic_n == dynamic_count
            and dynamic_count <= 36
        )

        fallback_n = 3
        if (
            "disrupts the pattern" in composite
            or "doesn't belong" in composite
            or "does not belong" in composite
            or "odd one out" in composite
            or "out of place" in composite
            or task_count >= 15
        ):
            fallback_n = 4
        elif task_count >= 9:
            fallback_n = 3

        use_dynamic = dynamic_square_ok and dynamic_count >= 9
        logger.info(
            "[Browser][Manual] tile infer: dynamic=%s dyn_n=%s task_count=%s fallback_n=%s use_dynamic=%s",
            dynamic_count,
            dynamic_n,
            task_count,
            fallback_n,
            use_dynamic,
        )

        # 若 DOM 提取失败/不可信，回退到网格切分（支持 3x3/4x4）
        if use_dynamic:
            tiles = dynamic_tiles
        else:
            # 非规则网格题（如“根据示例图识别”）会出现散点目标（常见 6 个对象），
            # DOM 选择器经常只能拿到 2~4 个元素；此时尝试从截图做一次轻量目标中心检测。
            tiles = []
            if box and dynamic_count < 6:
                try:
                    from PIL import Image
                    import numpy as np
                    from scipy import ndimage as ndi

                    os.makedirs("test_outputs", exist_ok=True)
                    vision_shot = f"test_outputs/.tmp_hcaptcha_detect_{int(time.time() * 1000)}.png"
                    try:
                        page.screenshot(path=vision_shot)
                    except Exception:
                        vision_shot = ""
                    if not vision_shot or (not os.path.exists(vision_shot)):
                        raise RuntimeError("vision_shot_unavailable")

                    page_img = np.array(Image.open(vision_shot).convert("RGB"))
                    px = int(max(0, box.get("x", 0)))
                    py = int(max(0, box.get("y", 0)))
                    pw = int(max(0, box.get("width", 0)))
                    ph = int(max(0, box.get("height", 0)))
                    if pw > 120 and ph > 120:
                        # 只分析 challenge 下半区（标题/示例区域以外），降低误检。
                        ax0 = px + int(pw * 0.04)
                        ax1 = px + int(pw * 0.96)
                        ay0 = py + int(ph * 0.30)
                        ay1 = py + int(ph * 0.92)
                        area = page_img[max(0, ay0):min(page_img.shape[0], ay1), max(0, ax0):min(page_img.shape[1], ax1)]
                        if area.size > 0:
                            gray = (0.299 * area[:, :, 0] + 0.587 * area[:, :, 1] + 0.114 * area[:, :, 2]).astype(np.float32)
                            blur = ndi.gaussian_filter(gray, sigma=4.2)
                            high = np.abs(gray - blur)
                            thr = float(np.percentile(high, 86))
                            mask = high > max(14.0, thr)
                            mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
                            mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))

                            lbl, nlab = ndi.label(mask)
                            objs = ndi.find_objects(lbl)
                            cand = []
                            for i, s in enumerate(objs, start=1):
                                if s is None:
                                    continue
                                ys, xs = s
                                hh = int(ys.stop - ys.start)
                                ww = int(xs.stop - xs.start)
                                if hh < 20 or ww < 20:
                                    continue
                                comp = (lbl[s] == i)
                                area_px = int(comp.sum())
                                if area_px < 260 or area_px > 5500:
                                    continue
                                cy, cx = ndi.center_of_mass(comp)
                                abs_x = int(ax0 + xs.start + cx)
                                abs_y = int(ay0 + ys.start + cy)
                                # 去重
                                dup = False
                                for ox, oy in cand:
                                    if abs(abs_x - ox) < 18 and abs(abs_y - oy) < 18:
                                        dup = True
                                        break
                                if not dup:
                                    cand.append((abs_x, abs_y))

                            if 5 <= len(cand) <= 12:
                                cand.sort(key=lambda p: (p[1], p[0]))
                                tiles = [(x - px, y - py) for (x, y) in cand]
                                logger.info("[Browser][Manual] 视觉检测候选点: %s", len(tiles))
                except Exception:
                    tiles = []

            if not tiles:
            # 4x4 pattern 题在当前 Stripe/hCaptcha 版本中经常拿到“整卡片”grid_rect（包含标题区），
            # 会导致第一行坐标落在 prompt 区。该场景优先回退到 frame 经验比例。
                force_frame_ratio = bool(fallback_n >= 4 and dynamic_count < 9)

                if (not force_frame_ratio) and isinstance(grid_rect, dict) and int(grid_rect.get("w") or 0) > 140 and int(grid_rect.get("h") or 0) > 140:
                    gx = int(grid_rect.get("x") or 0)
                    gy = int(grid_rect.get("y") or 0)
                    gw = int(grid_rect.get("w") or 0)
                    gh = int(grid_rect.get("h") or 0)

                    # 某些题面中 grid_rect 会落在整张 challenge 卡片（含标题/底栏）而非纯图片网格。
                    # 当高度明显偏高时，裁掉上方标题区和下方 footer，聚焦中间图片区。
                    # 经验比例来自 Stripe hCaptcha challenge 布局。
                    if gh > int(gw * 1.06):
                        top_trim = int(gh * 0.22)
                        bottom_trim = int(gh * 0.18)
                        if gh - top_trim - bottom_trim > 120:
                            gy += top_trim
                            gh -= (top_trim + bottom_trim)
                    x0, x1 = gx + int(gw * 0.06), gx + int(gw * 0.94)
                    y0, y1 = gy + int(gh * 0.06), gy + int(gh * 0.94)
                else:
                    x0, x1 = int(width * 0.12), int(width * 0.74)
                    y0, y1 = int(height * 0.24), int(height * 0.73)

                n = max(2, min(6, int(fallback_n or 3)))
                xs = [int(x0 + (x1 - x0) * (i + 0.5) / n) for i in range(n)]
                ys = [int(y0 + (y1 - y0) * (i + 0.5) / n) for i in range(n)]
                tiles = [(x, y) for y in ys for x in xs]  # row-major

        # 拖拽题的落点通常是连续区域（非离散格子）。
        # 用可视区域网格增强候选点，避免仅依赖弱视觉分割导致“永远拖不到正确区域”。
        if is_drag_challenge:
            try:
                scene = frame.evaluate(
                    """
                    () => {
                      const els = Array.from(document.querySelectorAll('canvas, img, svg, .task-image, .task'));
                      let best = null;
                      for (const el of els) {
                        const r = el.getBoundingClientRect();
                        if (r.width < 180 || r.height < 150) continue;
                        const cs = getComputedStyle(el);
                        if (cs.display === 'none' || cs.visibility === 'hidden') continue;
                        const area = r.width * r.height;
                        if (!best || area > best.area) {
                          best = {x: r.left, y: r.top, w: r.width, h: r.height, area};
                        }
                      }
                      return best || null;
                    }
                    """
                )
            except Exception:
                scene = None

            dense = []
            if isinstance(scene, dict):
                try:
                    sx = int(scene.get("x") or 0)
                    sy = int(scene.get("y") or 0)
                    sw = int(scene.get("w") or 0)
                    sh = int(scene.get("h") or 0)
                    if sw > 180 and sh > 150:
                        try:
                            cols = max(3, min(8, int(os.getenv("HCAPTCHA_DRAG_DENSE_COLS", "4"))))
                        except Exception:
                            cols = 4
                        try:
                            rows = max(2, min(6, int(os.getenv("HCAPTCHA_DRAG_DENSE_ROWS", "3"))))
                        except Exception:
                            rows = 3
                        for ry in range(rows):
                            for rx in range(cols):
                                px = sx + int(sw * (0.1 + 0.8 * (rx / max(1, cols - 1))))
                                py = sy + int(sh * (0.12 + 0.76 * (ry / max(1, rows - 1))))
                                dense.append((px, py))
                except Exception:
                    dense = []

            if dense:
                merged = []
                for x, y in (tiles + dense):
                    dup = False
                    for ox, oy in merged:
                        if abs(x - ox) < 16 and abs(y - oy) < 16:
                            dup = True
                            break
                    if not dup:
                        merged.append((x, y))
                try:
                    max_drag_candidates = max(8, min(36, int(os.getenv("HCAPTCHA_DRAG_MAX_CANDIDATES", "16"))))
                except Exception:
                    max_drag_candidates = 16
                tiles = merged[:max_drag_candidates]
                logger.info("[Browser][Manual] 拖拽候选增强: base+dense=%s", len(tiles))
        if not box:
            return False

        # 导出当前页面截图
        os.makedirs("test_outputs", exist_ok=True)
        ts = int(time.time() * 1000)
        shot_path = f"test_outputs/manual_hcaptcha_{ts}.png"
        try:
            page.screenshot(path=shot_path)
        except Exception:
            shot_path = ""

        prompt_show = (prompt or "").strip() or "(无 prompt)"
        logger.info("[Browser][Manual] prompt=%s submit=%s frame=%s", prompt_show, submit_text or "", (frame_url or "")[:120])
        if shot_path:
            logger.info("[Browser][Manual] 截图已保存: %s", shot_path)
        logger.info("[Browser][Manual] 可点击格数: %s", len(tiles))
        if selector_debug:
            logger.info("[Browser][Manual] selector counts: %s", selector_debug)
        if task_assets.get("examples") or task_assets.get("tasks"):
            logger.info(
                "[Browser][Manual] asset urls: examples=%s tasks=%s",
                len(task_assets.get("examples") or []),
                len(task_assets.get("tasks") or []),
            )

        if os.getenv("HCAPTCHA_DUMP_DOM", "0") not in ("0", "false", "False"):
            try:
                dom_dump = frame.evaluate(
                    """
                    () => {
                      const nodes = Array.from(document.querySelectorAll('.task, .task-image, [data-task-key], [taskkey], [class*="task"]'));
                      return nodes.slice(0, 40).map((el) => {
                        const attrs = {};
                        for (const a of Array.from(el.attributes || [])) {
                          attrs[a.name] = String(a.value || '').slice(0, 180);
                        }
                        const r = el.getBoundingClientRect();
                        return {
                          tag: el.tagName,
                          cls: String(el.className || '').slice(0, 160),
                          text: String(el.innerText || '').slice(0, 120),
                          attrs,
                          rect: {x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height)},
                        };
                      });
                    }
                    """
                ) or []
                if isinstance(dom_dump, list):
                    os.makedirs("test_outputs", exist_ok=True)
                    ts2 = int(time.time() * 1000)
                    p2 = f"test_outputs/hcaptcha_domdump_{ts2}.json"
                    with open(p2, "w", encoding="utf-8") as f:
                        json.dump(dom_dump, f, ensure_ascii=False, indent=2)
                    logger.info("[Browser][Manual] 已保存 DOM dump: %s (nodes=%s)", p2, len(dom_dump))
            except Exception:
                pass

        # 生成带编号的辅助截图，方便人工输入索引。
        if shot_path and box and tiles:
            try:
                from PIL import Image, ImageDraw

                im = Image.open(shot_path).convert("RGBA")
                draw = ImageDraw.Draw(im)
                # 估算点位间距，把标签尽量放在点位左上，降低遮挡主体目标。
                nearest = 38
                try:
                    dists = []
                    for i, (ax, ay) in enumerate(tiles):
                        for j, (bx, by) in enumerate(tiles):
                            if j <= i:
                                continue
                            dd = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                            if dd > 8:
                                dists.append(dd)
                    if dists:
                        nearest = int(max(24, min(96, sorted(dists)[0])))
                except Exception:
                    nearest = 38
                off = int(max(10, min(26, nearest * 0.34)))
                for idx, (tx, ty) in enumerate(tiles, start=1):
                    px = int(box["x"] + tx)
                    py = int(box["y"] + ty)
                    lx = px - off
                    ly = py - off
                    draw.rectangle(
                        (lx - 8, ly - 8, lx + 8, ly + 8),
                        fill=(0, 0, 0, 190),
                        outline=(255, 255, 255, 220),
                        width=1,
                    )
                    draw.text((lx - 4, ly - 7), str(idx), fill=(255, 255, 255, 255))
                indexed_path = shot_path.replace(".png", "_indexed.png")
                im.convert("RGB").save(indexed_path)
                logger.info("[Browser][Manual] 编号截图: %s", indexed_path)
            except Exception:
                pass

        # 允许的输入:
        # - "1,4,7" 选择格子
        # - 空串: 不点格子直接提交
        # - "r": 尝试刷新 challenge
        # - "q": 跳过本轮（不中断主流程）
        auto_mode_enabled = os.getenv("HCAPTCHA_AUTO_PATTERN_LLM", "0") not in ("0", "false", "False")
        drag_prompt_hint = any(
            k in (prompt_show or "").lower()
            for k in (
                "drag the segment",
                "complete the line",
                "drag the piece",
                "drag the object",
                "matching shadow",
                "slide the piece",
            )
        )

        min_i = 0
        max_i = 0
        try:
            min_c = req_cfg.get("multiple_choice_min_choices")
            max_c = req_cfg.get("multiple_choice_max_choices")
            min_i = int(min_c) if min_c is not None else 0
            max_i = int(max_c) if max_c is not None else 0
        except Exception:
            min_i = 0
            max_i = 0

        cfg_force_single = False
        # 注意：area_select 常见 max_choices=1 但 tasklist 会有多项，
        # 此时“单选”是每张图单选，不代表整题只点一次，不能误判为全局单选。
        if is_binary_mode and max_i == 1 and min_i <= 1:
            cfg_force_single = True
        elif (not is_area_select_mode) and max_i == 1 and min_i <= 1:
            cfg_force_single = True
        if not cfg_force_single:
            p_lc = (prompt_show or "").lower()
            if (
                ("sample image" in p_lc or "example image" in p_lc)
                and (
                    "identify the one" in p_lc
                    or "choose everything" in p_lc
                    or "identify everything" in p_lc
                )
            ):
                cfg_force_single = True
        allow_drag_llm = os.getenv("HCAPTCHA_DRAG_LLM", "1") not in ("0", "false", "False")
        if cfg_force_single:
            logger.info("[Browser][Manual] request_config 指示单选题 (max_choices=1)")

        # area_select: 通常 tasklist 中每个 task_key 需要各自一个点。
        required_clicks = 1
        if is_area_select_mode:
            try:
                required_clicks = max(1, int(os.getenv("HCAPTCHA_AREA_REQUIRED_CLICKS", "1")))
            except Exception:
                required_clicks = 1
            # getcaptcha 已给出 tasklist 时，至少要覆盖每个 task_key 一次，
            # 否则 checkcaptcha 经常会出现某些 key 空数组并被 400 拒绝。
            if net_task_count > 0:
                required_clicks = max(required_clicks, int(net_task_count))
            required_clicks = max(1, min(len(tiles) or 1, required_clicks))

        auto_picks: list[int] = []
        if shot_path and tiles and auto_mode_enabled and ((not drag_prompt_hint) or allow_drag_llm):
            # 优先用带编号截图给 LLM，避免标号与题图错位。
            auto_img = shot_path.replace(".png", "_indexed.png")
            if os.path.exists(auto_img):
                auto_picks = self._solve_pattern_with_llm(
                    auto_img,
                    len(tiles),
                    prompt=prompt_show,
                    force_single=cfg_force_single,
                )
            else:
                auto_picks = self._solve_pattern_with_llm(
                    shot_path,
                    len(tiles),
                    prompt=prompt_show,
                    force_single=cfg_force_single,
                )

        # 单选题：按候选轮询，避免 AutoLLM 连续命中同一个错误点位。
        single_prompt_hint = any(
            k in (prompt_show or "").lower()
            for k in (
                "identify the one",
                "matching silhouette",
                "line breaks",
                "click on the point",
                "disrupts the pattern",
                "odd one out",
                "select a language",
                "choose a language",
            )
        )
        frame_base = (frame_url or "").split("#")[0]
        single_mode = bool((cfg_force_single or single_prompt_hint) and (not is_area_select_mode))
        if auto_mode_enabled and single_mode and tiles:
            prefer = []
            for idx1 in auto_picks:
                try:
                    z = int(idx1) - 1
                except Exception:
                    z = -1
                if 0 <= z < len(tiles):
                    prefer.append(z)
            base_order = list(range(len(tiles)))
            if prefer:
                # 仅首次把 LLM 首选前置；后续按轮询顺序遍历全部候选。
                first = prefer[0]
                if first in base_order:
                    base_order.remove(first)
                    base_order.insert(0, first)
            key = f"{frame_base}::single::{prompt_show}::{len(tiles)}"
            st = self._single_try_state.get(key)
            if not isinstance(st, dict) or not isinstance(st.get("order"), list) or len(st.get("order") or []) != len(base_order):
                st = {"order": base_order, "pos": 0}
                self._single_try_state[key] = st
            order = st.get("order") or base_order
            pos = int(st.get("pos", 0))
            if order:
                chosen = order[pos % len(order)]
                st["pos"] = pos + 1
                auto_picks = [chosen + 1]
                logger.info(
                    "[Browser][Manual] 单选轮询候选: chosen=%s order=%s pos=%s",
                    chosen + 1,
                    [x + 1 for x in order],
                    st["pos"],
                )

        # area_select: 尽量保证每个 task_key 都有候选点，避免 checkcaptcha 出现空数组。
        if auto_mode_enabled and is_area_select_mode and tiles:
            prefer = []
            for idx1 in auto_picks:
                try:
                    z = int(idx1) - 1
                except Exception:
                    z = -1
                if 0 <= z < len(tiles) and z not in prefer:
                    prefer.append(z)
            order = []
            for z in prefer + list(range(len(tiles))):
                if z not in order:
                    order.append(z)
            key = f"{frame_base}::area::{prompt_show}::{len(tiles)}::{required_clicks}"
            st = self._area_try_state.get(key)
            if (
                not isinstance(st, dict)
                or not isinstance(st.get("order"), list)
                or len(st.get("order") or []) != len(order)
            ):
                st = {"order": order, "pos": 0}
                self._area_try_state[key] = st
            order = st.get("order") or order
            pos = int(st.get("pos", 0))
            picks0 = []
            if order:
                for off in range(len(order)):
                    cand = order[(pos + off) % len(order)]
                    if cand not in picks0:
                        picks0.append(cand)
                    if len(picks0) >= max(1, required_clicks):
                        break
                st["pos"] = pos + max(1, required_clicks)
                auto_picks = [z + 1 for z in picks0]
                logger.info(
                    "[Browser][Manual] 区域点选轮询: chosen=%s need=%s pos=%s",
                    auto_picks,
                    required_clicks,
                    st["pos"],
                )

        # 自动模式兜底：尽量避免“空提交”。
        if auto_mode_enabled and (not auto_picks) and tiles:
            try:
                fallback_picks = max(1, min(len(tiles), int(os.getenv("HCAPTCHA_AUTO_FALLBACK_PICKS", "1"))))
            except Exception:
                fallback_picks = 1
            if is_area_select_mode:
                auto_picks = list(range(1, min(len(tiles), max(1, required_clicks)) + 1))
            elif single_mode:
                auto_picks = [1]
            else:
                # binary/多选题也避免空提交；默认更保守，避免一次性误选过多。
                auto_picks = list(range(1, fallback_picks + 1))

        if auto_picks:
            user_raw = ",".join(str(i) for i in auto_picks)
            logger.info("[Browser][Manual] AutoLLM 输入: %s", user_raw)
        else:
            try:
                user_raw = input(
                    f"[HCAPTCHA MANUAL] 输入要点击的格子编号(1-{max(1, len(tiles))},逗号分隔；空=仅提交；r=刷新；q=跳过): "
                ).strip().lower()
            except Exception as e:
                if auto_mode_enabled:
                    if tiles:
                        try:
                            fallback_picks = max(1, min(len(tiles), int(os.getenv("HCAPTCHA_AUTO_FALLBACK_PICKS", "1"))))
                        except Exception:
                            fallback_picks = 1
                        user_raw = ",".join(str(i) for i in range(1, fallback_picks + 1))
                        logger.warning(
                            "[Browser][Manual] 读取输入失败: %s (自动模式回退兜底索引=%s)",
                            e,
                            user_raw,
                        )
                    else:
                        logger.warning("[Browser][Manual] 读取输入失败: %s (自动模式回退为空提交)", e)
                        user_raw = ""
                else:
                    logger.warning("[Browser][Manual] 读取输入失败: %s", e)
                    return False

        if user_raw == "q":
            logger.info("[Browser][Manual] 用户选择跳过本轮")
            return False

        if user_raw == "r":
            for sel in [
                ".refresh",
                ".button-reload",
                "[aria-label*='refresh']",
                "[title*='refresh' i]",
            ]:
                try:
                    ok = frame.evaluate(
                        """
                        (s) => {
                          const el = document.querySelector(s);
                          if (!el) return false;
                          el.click();
                          return true;
                        }
                        """,
                        sel,
                    )
                    if ok:
                        logger.info("[Browser][Manual] 已触发 refresh")
                        return True
                except Exception:
                    continue
            logger.warning("[Browser][Manual] 未找到 refresh 按钮")
            return False

        indices = []
        if user_raw:
            parts = [p.strip() for p in user_raw.replace("，", ",").split(",") if p.strip()]
            for p in parts:
                if p.isdigit():
                    idx1 = int(p)
                    if 1 <= idx1 <= len(tiles):
                        z = idx1 - 1
                        if z not in indices:
                            indices.append(z)

        # area_select: 需要尽量补齐每个 task 的点位，避免部分 task_key 为空。
        if is_area_select_mode and tiles and len(indices) < max(1, required_clicks):
            key = f"{frame_base}::area::{prompt_show}::{len(tiles)}::{required_clicks}"
            st = self._area_try_state.get(key)
            order = []
            if isinstance(st, dict) and isinstance(st.get("order"), list):
                order = [int(x) for x in st.get("order", []) if isinstance(x, int)]
            if not order:
                order = list(range(len(tiles)))
            pos = int(st.get("pos", 0)) if isinstance(st, dict) else 0
            for off in range(len(order)):
                cand = order[(pos + off) % len(order)]
                if 0 <= cand < len(tiles) and cand not in indices:
                    indices.append(cand)
                if len(indices) >= max(1, required_clicks):
                    break
            if isinstance(st, dict):
                st["pos"] = pos + max(1, required_clicks)
            logger.info(
                "[Browser][Manual] 区域点选补全: indices=%s need=%s",
                [i + 1 for i in indices],
                required_clicks,
            )

        # 拖拽类题目（例如 "Please drag the segment to complete the line"）:
        # 需要从 "+ Move" 位置拖到目标点，且通常没有 Verify 按钮。
        drag_handle = None
        if is_drag_challenge:
            try:
                drag_handle = frame.evaluate(
                    """
                    () => {
                      const all = Array.from(document.querySelectorAll('button, div, span, a'));
                      const cands = [];
                      for (const el of all) {
                        const txt = ((el.innerText || el.textContent || '') + ' ' + (el.getAttribute('aria-label') || '')).toLowerCase();
                        if (!txt.includes('move')) continue;
                        const r = el.getBoundingClientRect();
                        if (r.width < 18 || r.height < 12) continue;
                        const cs = getComputedStyle(el);
                        if (cs.display === 'none' || cs.visibility === 'hidden') continue;
                        cands.push({
                          x: r.left + r.width / 2,
                          y: r.top + r.height / 2,
                          w: r.width,
                          h: r.height,
                          left: r.left,
                          top: r.top,
                          area: r.width * r.height,
                          txt,
                        });
                      }
                      if (!cands.length) return null;
                      cands.sort((a, b) => b.area - a.area);
                      const c = cands[0];
                      return {
                        x: Math.round(c.x),
                        y: Math.round(c.y),
                        w: Math.round(c.w),
                        h: Math.round(c.h),
                        left: Math.round(c.left),
                        top: Math.round(c.top),
                        txt: c.txt || '',
                      };
                    }
                    """
                )
            except Exception:
                drag_handle = None
            logger.info(
                "[Browser][Manual] 检测到拖拽题: drag_handle=%s",
                drag_handle if isinstance(drag_handle, dict) else None,
            )

            # AutoLLM 在拖拽题上波动较大：为避免一直重复同一点，启用“按候选点轮询”。
            # 顺序: AutoLLM 首选 -> 其余候选按索引顺序。
            if auto_mode_enabled:
                prefer = [i for i in indices if 0 <= i < len(tiles)]
                base = list(range(len(tiles)))
                order = []
                for i in prefer + base:
                    if i not in order:
                        order.append(i)
                drag_key = f"{(frame_url or '').split('#')[0]}::{prompt_show}::{len(tiles)}::{tiles[:3]}"
                st = self._drag_try_state.get(drag_key)
                if not isinstance(st, dict) or st.get("order") != order:
                    st = {"order": order, "pos": 0}
                    self._drag_try_state[drag_key] = st
                pos = int(st.get("pos", 0))
                if order:
                    chosen = order[pos % len(order)]
                    st["pos"] = pos + 1
                    indices = [chosen]
                    logger.info(
                        "[Browser][Manual] 拖拽轮询候选: chosen=%s order=%s pos=%s",
                        chosen + 1,
                        [x + 1 for x in order],
                        st["pos"],
                    )

        clicked = 0
        if is_drag_challenge and indices:
            i = indices[0]
            if 0 <= i < len(tiles):
                tx, ty = tiles[i]
                # 拖拽起点优先使用 "+Move" 句柄；若句柄不可见，则回退到题图区右上角经验点位。
                start_x = box["x"] + int(max(20, width * 0.86))
                start_y = box["y"] + int(max(20, height * 0.33))
                if isinstance(drag_handle, dict):
                    hx = int(drag_handle.get("x") or 0)
                    hy = int(drag_handle.get("y") or 0)
                    hw = int(drag_handle.get("w") or 0)
                    hh = int(drag_handle.get("h") or 0)
                    hl = int(drag_handle.get("left") or 0)
                    ht = int(drag_handle.get("top") or 0)
                    if hw >= 40 and hh >= 24 and hl >= 0 and ht >= 0:
                        # 文案通常在块上方，真实可拖拽片段位于下半区。
                        sx = hl + int(hw * 0.58)
                        sy = ht + int(max(8, hh * 0.72))
                        start_x = box["x"] + sx
                        start_y = box["y"] + sy
                    elif hx > 0 and hy > 0:
                        start_x = box["x"] + hx
                        start_y = box["y"] + hy
                end_x = box["x"] + tx
                end_y = box["y"] + ty
                try:
                    page.mouse.move(start_x, start_y)
                    page.mouse.down()
                    time.sleep(0.18)
                    page.mouse.move(end_x, end_y, steps=24)
                    time.sleep(0.08)
                    page.mouse.up()
                    clicked = 1
                    logger.info(
                        "[Browser][Manual] 拖拽执行: from=(%s,%s) to=(%s,%s) idx=%s",
                        int(start_x),
                        int(start_y),
                        int(end_x),
                        int(end_y),
                        i + 1,
                    )
                except Exception as drag_err:
                    logger.warning("[Browser][Manual] 拖拽失败，回退 click: %s", drag_err)
                    try:
                        page.mouse.click(end_x, end_y, delay=35)
                        clicked = 1
                    except Exception:
                        pass
        else:
            for i in indices:
                if i < 0 or i >= len(tiles):
                    continue
                x, y = tiles[i]
                try:
                    # 优先使用坐标点击，避免 DOM 顺序与视觉顺序不一致。
                    page.mouse.click(box["x"] + x, box["y"] + y, delay=35)
                    clicked += 1
                    continue
                except Exception:
                    pass

                # 坐标点击失败时，回退到按 aria-label 序号点击。
                try:
                    ok = frame.evaluate(
                        """
                        (idx1) => {
                          const sel = `.task[aria-label*='${idx1}'], .task-image[aria-label*='${idx1}']`;
                          const el = document.querySelector(sel);
                          if (!el) return false;
                          el.click();
                          return true;
                        }
                        """,
                        i + 1,
                    )
                    if ok:
                        clicked += 1
                except Exception:
                    continue

        # 提交：优先 Verify/Next，再退化到 Skip/第一个按钮
        submitted = False
        picked_submit = ""
        # 拖拽题通常只有 Skip/Next，若禁用 skip 会卡在同一题面循环。
        allow_skip_submit = True
        # area_select / 必选题（min_choices>=1）禁用 skip 兜底，避免发空答案。
        if is_area_select_mode:
            allow_skip_submit = False
        try:
            submit_result = frame.evaluate(
                """
                (opts) => {
                  const allowSkip = !!(opts && opts.allowSkip);
                  const all = Array.from(document.querySelectorAll('button, .button, .button-submit'))
                    .filter(el => {
                      const r = el.getBoundingClientRect();
                      if (r.width < 20 || r.height < 20) return false;
                      const cs = getComputedStyle(el);
                      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
                      return true;
                    });
                  if (!all.length) return { ok: false, picked: "" };

                  const getText = (el) => ((el.innerText || el.getAttribute('aria-label') || "").trim().toLowerCase());
                  let picked = null;
                  for (const el of all) {
                    const t = getText(el);
                    if (t.includes('verify') || t.includes('next')) { picked = el; break; }
                  }
                  if (!picked && allowSkip) {
                    for (const el of all) {
                      const t = getText(el);
                      if (t.includes('skip')) { picked = el; break; }
                    }
                  }
                  if (!picked) return { ok: false, picked: "" };
                  const txt = getText(picked);
                  picked.click();
                  return { ok: true, picked: txt };
                }
                """,
                {"allowSkip": allow_skip_submit},
            ) or {}
            submitted = bool(submit_result.get("ok"))
            picked_submit = submit_result.get("picked", "")
        except Exception:
            submitted = False

        if clicked or submitted:
            logger.info(
                "[Browser][Manual] 本轮完成: clicked=%s indices=%s submitted=%s submit=%s",
                clicked,
                [i + 1 for i in indices],
                submitted,
                picked_submit,
            )
            return True
        return False

    def _attempt_visual_challenge(self, page, frame, frame_url: str) -> bool:
        """
        对 hCaptcha 图像题执行一轮启发式点击。

        重点改进：
        1) 优先基于 challenge 网格（canvas/task-grid）定位 3x3 中心，避免点到外层容器；
        2) challenge 已出现后不再无脑点 checkbox；
        3) 遇到 “Please try again” 状态优先尝试 refresh，而不是盲点网格。
        """
        import os

        info = frame.evaluate(
            """
            () => {
              const text = (document.body && document.body.innerText) || "";
              const lines = text.split("\\n").map(s => s.trim()).filter(Boolean);
              let prompt = "";
              const promptLike = [];
              for (const line of lines) {
                if (!line) continue;
                if (
                  /^(tap on|select|click all|choose everything|please drag|drag|identify|pick|choose all)/i.test(line) ||
                  /disrupts\\s+the\\s+pattern/i.test(line) ||
                  /complete\\s+the\\s+line/i.test(line) ||
                  /doesn['’]t\\s+belong/i.test(line) ||
                  /odd\\s+one\\s+out/i.test(line) ||
                  /identify everything/i.test(line) ||
                  /example image/i.test(line)
                ) {
                  promptLike.push(line);
                }
              }

              // DOM 层再找一次，防止 innerText 抽不到（某些 hCaptcha 版本用 aria/特殊容器）。
              const qSels = [
                "[class*='prompt']",
                "[class*='question']",
                "[class*='instruction']",
                "[data-testid*='prompt']",
                "[data-testid*='question']",
                "[aria-live]",
                "[role='heading']",
              ];
              for (const s of qSels) {
                for (const el of Array.from(document.querySelectorAll(s))) {
                  const t = ((el.innerText || el.textContent || "") + " " + (el.getAttribute('aria-label') || "")).trim();
                  if (!t) continue;
                  if (t.length > 260) continue;
                  if (
                    /select|click|choose|drag|identify|pattern|example image|belong|odd one/i.test(t)
                  ) {
                    promptLike.push(t.replace(/\\s+/g, " ").trim());
                  }
                }
              }
              if (promptLike.length) {
                prompt = promptLike
                  .filter(Boolean)
                  .sort((a, b) => b.length - a.length)[0] || "";
              }

              const hasTryAgain = /please\\s+try\\s+again/i.test(text);

              const submitSel = [
                ".button-submit.button",
                "button[type='submit']",
                "button[aria-label*='Verify']",
                "button[aria-label*='Next']",
                "button[aria-label*='Skip']",
                ".button-submit",
              ];
              let submitText = "";
              for (const s of submitSel) {
                const el = document.querySelector(s);
                if (el) {
                  submitText = (el.innerText || el.getAttribute('aria-label') || "").trim();
                  break;
                }
              }

              const taskCount = document.querySelectorAll('.task-image, .task').length;

              // 优先找真实网格区域：
              // 1) .task-grid / .challenge-container 内的大画布
              // 2) 兜底用 challenge 容器经验比例
              function toRect(r) {
                if (!r) return null;
                return {
                  x: Math.max(0, Math.floor(r.left)),
                  y: Math.max(0, Math.floor(r.top)),
                  w: Math.max(0, Math.floor(r.width)),
                  h: Math.max(0, Math.floor(r.height)),
                };
              }

              let gridRect = null;
              const gridLike = document.querySelector('.task-grid') || document.querySelector('.challenge-container');
              if (gridLike) {
                const r = gridLike.getBoundingClientRect();
                if (r.width > 160 && r.height > 160) {
                  gridRect = toRect(r);
                }
              }

              const canvases = Array.from(document.querySelectorAll('canvas'))
                .map(el => ({el, r: el.getBoundingClientRect()}))
                .filter(x => x.r.width > 120 && x.r.height > 120);
              if (canvases.length) {
                // 选 canvas 时避免直接拿“最大面积”：
                // 某些版本里最大 canvas 是整张 challenge 卡片（含标题+底栏），
                // 会导致网格切分严重偏移。这里对“过高/过宽”的候选做惩罚，
                // 并适度偏向位于中下部的 canvas（通常是真实题图）。
                let best = null;
                for (const c of canvases) {
                  const r = c.r;
                  const area = Math.max(1, r.width * r.height);
                  const aspect = r.width / Math.max(1, r.height);
                  let score = area;
                  if (r.height > (window.innerHeight * 0.72)) score *= 0.52;
                  if (r.width > (window.innerWidth * 0.92)) score *= 0.75;
                  if (aspect < 0.7 || aspect > 1.8) score *= 0.8;
                  if (r.top > (window.innerHeight * 0.20)) score *= 1.28;
                  if (r.top > (window.innerHeight * 0.32)) score *= 1.16;
                  if (!best || score > best.score) {
                    best = { score, r };
                  }
                }
                if (best && best.r) {
                  const canvasRect = toRect(best.r);
                  if (canvasRect && canvasRect.w > 160 && canvasRect.h > 160) {
                    // canvas 往往更贴近实际 3x3/4x4 图片区，优先使用
                    gridRect = canvasRect;
                  }
                }
              }

              // 尝试提取画布像素签名（用于识别是否切换到新题面）
              let imageSig = "";
              try {
                if (canvases.length) {
                  const c = canvases[0].el;
                  const ctx = c.getContext("2d");
                  if (ctx) {
                    const pts = [
                      [0.15, 0.15], [0.5, 0.15], [0.85, 0.15],
                      [0.15, 0.5],  [0.5, 0.5],  [0.85, 0.5],
                      [0.15, 0.85], [0.5, 0.85], [0.85, 0.85],
                    ];
                    const vals = [];
                    for (const [rx, ry] of pts) {
                      const x = Math.max(0, Math.min(c.width - 1, Math.floor(c.width * rx)));
                      const y = Math.max(0, Math.min(c.height - 1, Math.floor(c.height * ry)));
                      const p = ctx.getImageData(x, y, 1, 1).data;
                      vals.push(`${p[0]}-${p[1]}-${p[2]}`);
                    }
                    imageSig = vals.join("|");
                  }
                }
              } catch (e) {}

              const selectedHints = Array.from(document.querySelectorAll("*"))
                .filter(el => {
                  const cls = (el.className || "").toString().toLowerCase();
                  if (!cls) return false;
                  return cls.includes("selected") || cls.includes("active") || cls.includes("answer");
                })
                .slice(0, 40)
                .map(el => `${el.tagName}#${el.id || ''}.${(el.className || '').toString().slice(0, 60)}`);

              return {
                prompt,
                submitText,
                taskCount,
                width: Math.max(0, window.innerWidth || 0),
                height: Math.max(0, window.innerHeight || 0),
                bodyText: text.slice(0, 1400),
                hasTryAgain,
                gridRect,
                selectedHints,
                imageSig,
              };
            }
            """
        ) or {}

        frame_base = frame_url.split('#')[0]
        prompt = (info.get("prompt") or "").strip()
        if prompt:
            self._last_visual_prompt[frame_base] = prompt
        else:
            prompt = (self._last_visual_prompt.get(frame_base, "") or "").strip()
        if not prompt:
            prompt = (self._last_net_prompt or "").strip()
            if prompt:
                self._last_visual_prompt[frame_base] = prompt
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        grid_rect = info.get("gridRect") or {}
        has_try_again = bool(info.get("hasTryAgain"))
        task_count = int(info.get("taskCount") or 0)
        body_text_lc = (info.get("bodyText") or "").strip().lower()

        # 没有可用尺寸时放弃此轮。
        if width < 200 or height < 200:
            return False

        # 某些 challenge 轮次会退化成 “Select the checkbox below / I am human” 小卡片。
        # 该状态下优先直接点 checkbox，而不是进入九宫格流程。
        if (
            "select the checkbox below" in body_text_lc
            or "i am human" in body_text_lc
            or "one more step before you're done" in body_text_lc
        ):
            for sel in ["#checkbox", "[role='checkbox']", "[id*='checkbox']"]:
                try:
                    el = frame.query_selector(sel)
                    if el:
                        el.click(force=True, timeout=1200)
                        logger.info("[Browser] hCaptcha checkbox-state 点击成功: %s", sel)
                        return True
                except Exception:
                    continue

        manual_mode = os.getenv("HCAPTCHA_MANUAL", "0") not in ("0", "false", "False")
        if manual_mode:
            # 图像尚未加载完整时，不要过早进入人工选择（避免一直对着 spinner）
            # 进入人工模式的条件：
            # 1) 已有明确 prompt；或
            # 2) 出现 3x3 图像任务（task_count>=9）
            ready_for_manual = bool(prompt) or task_count >= 9
            if (not ready_for_manual) and (not has_try_again):
                return False
            return self._manual_visual_step(
                page=page,
                frame=frame,
                frame_url=frame_url,
                prompt=prompt,
                width=width,
                height=height,
                grid_rect=grid_rect if isinstance(grid_rect, dict) else {},
                submit_text=(info.get("submitText") or ""),
                task_count=task_count,
                body_text=(info.get("bodyText") or ""),
            )

        # 以 (题面签名 + prompt + frame_url) 跟踪状态：
        # - attempt: 第几轮尝试
        # - mask: 当前已选中的 9bit 状态（我们维护的“点击后预期状态”）
        # 不直接把 imageSig 纳入 key（选中标记会改变像素，导致状态频繁重置）。
        # 以 frame + prompt 维持同题面的枚举连续性。
        key = f"{frame_base}::{prompt or 'no_prompt'}"
        state = self._challenge_attempt_state.get(key)
        if not isinstance(state, dict):
            state = {"attempt": 0, "mask": 0}
            self._challenge_attempt_state[key] = state
        attempt = int(state.get("attempt", 0))
        current_mask = int(state.get("mask", 0))
        state["attempt"] = attempt + 1

        # 若处于 "Please try again" 且尚未出现 prompt，通常是题面尚未渲染完成；
        # 这里不要频繁 refresh，避免一直卡在过渡态。
        if has_try_again and not prompt:
            if (attempt % 10) == 9:
                refreshed = False
                for sel in [
                    ".refresh",
                    ".button-reload",
                    "[aria-label*='refresh']",
                    "[title*='refresh' i]",
                ]:
                    try:
                        ok = frame.evaluate(
                            """
                            (s) => {
                              const el = document.querySelector(s);
                              if (!el) return false;
                              el.click();
                              return true;
                            }
                            """,
                            sel,
                        )
                        if ok:
                            refreshed = True
                            break
                    except Exception:
                        continue
                if refreshed:
                    state["mask"] = 0
                    logger.info("[Browser] hCaptcha challenge 处于 try-again 过渡态，周期性 refresh 一次")
                    return True
            return False

        # 3x3 栅格中心点。
        # 优先使用实际识别到的 gridRect，其次才用经验比例。
        if isinstance(grid_rect, dict) and int(grid_rect.get("w") or 0) > 140 and int(grid_rect.get("h") or 0) > 140:
            gx = int(grid_rect.get("x") or 0)
            gy = int(grid_rect.get("y") or 0)
            gw = int(grid_rect.get("w") or 0)
            gh = int(grid_rect.get("h") or 0)
            x0, x1 = gx + int(gw * 0.06), gx + int(gw * 0.94)
            y0, y1 = gy + int(gh * 0.06), gy + int(gh * 0.94)
        else:
            x0, x1 = int(width * 0.12), int(width * 0.74)
            y0, y1 = int(height * 0.24), int(height * 0.73)

        xs = [int(x0 + (x1 - x0) * (i + 0.5) / 3) for i in range(3)]
        ys = [int(y0 + (y1 - y0) * (i + 0.5) / 3) for i in range(3)]
        tiles = [(x, y) for y in ys for x in xs]

        # 9 位 Gray code 枚举（最多 511 组合），每步只翻转少量位，适合“持续试错”。
        # 通过 HCAPTCHA_ENUM_MAX 限制单题面最多尝试数，避免无限循环。
        try:
            enum_max = max(8, min(511, int(os.getenv("HCAPTCHA_ENUM_MAX", "220"))))
        except Exception:
            enum_max = 220
        enum_state = self._visual_enum_state.get(key)
        if not isinstance(enum_state, dict):
            seq = [((i ^ (i >> 1)) & 0x1FF) for i in range(1, 512)]
            enum_state = {"idx": 0, "seq": seq[:enum_max]}
            self._visual_enum_state[key] = enum_state

        seq = enum_state.get("seq") or [((i ^ (i >> 1)) & 0x1FF) for i in range(1, 1 + enum_max)]
        idx = int(enum_state.get("idx", 0))
        if idx >= len(seq):
            idx = 0
        target_mask = int(seq[idx]) & 0x1FF
        enum_state["idx"] = idx + 1
        indices = [i for i in range(9) if (target_mask >> i) & 1]

        mode = idx  # 复用日志位，表示枚举序号而非旧随机模式。

        # 仅点击 current_mask 与 target_mask 的差异位，避免“重复点击把已选中项取消”。
        diff_indices = [i for i in range(9) if ((current_mask ^ target_mask) >> i) & 1]
        click_points = [tiles[i] for i in diff_indices if 0 <= i < len(tiles)]

        # 使用 Playwright 真实鼠标点击（trusted event），避免 synthetic 事件被 hCaptcha 忽略。
        clicked = 0
        submitted = False
        hits = []

        try:
            frame_el = frame.frame_element()
            box = frame_el.bounding_box() if frame_el else None
        except Exception:
            box = None

        if box:
            for (x, y) in click_points:
                try:
                    hit = frame.evaluate(
                        """
                        ([px, py]) => {
                          const el = document.elementFromPoint(px, py);
                          if (!el) return "null";
                          return `${el.tagName || ''}#${el.id || ''}.${(el.className || '').toString().slice(0, 36)}`;
                        }
                        """,
                        [x, y],
                    )
                except Exception:
                    hit = "unknown"
                hits.append(hit)
                try:
                    page.mouse.click(box["x"] + x, box["y"] + y, delay=30)
                    clicked += 1
                except Exception:
                    continue

        # 更新本地状态：点击 diff 后预期达到 target_mask。
        state["mask"] = target_mask

        # 提交按钮：优先 Verify/Next，再退化到 Skip。
        try:
            submit_result = frame.evaluate(
                """
                () => {
                  const all = Array.from(document.querySelectorAll('button, .button, .button-submit'))
                    .filter(el => {
                      const r = el.getBoundingClientRect();
                      if (r.width < 20 || r.height < 20) return false;
                      const cs = getComputedStyle(el);
                      if (cs.display === 'none' || cs.visibility === 'hidden') return false;
                      return true;
                    });
                  if (!all.length) return { ok: false, picked: "" };

                  const getText = (el) => ((el.innerText || el.getAttribute('aria-label') || "").trim().toLowerCase());

                  let picked = null;
                  for (const el of all) {
                    const t = getText(el);
                    if (t.includes('verify') || t.includes('next')) { picked = el; break; }
                  }
                  if (!picked) {
                    for (const el of all) {
                      const t = getText(el);
                      if (t.includes('skip')) { picked = el; break; }
                    }
                  }
                  if (!picked) picked = all[0];
                  const txt = getText(picked);
                  picked.click();
                  return { ok: true, picked: txt };
                }
                """
            ) or {}
            submitted = bool(submit_result.get("ok"))
            picked_submit = submit_result.get("picked", "")
        except Exception:
            picked_submit = ""

        # 可选: 保存调试截图（默认关）。
        if os.getenv("HCAPTCHA_DEBUG_SHOTS", "0") not in ("0", "false", "False"):
            try:
                ts = int(time.time() * 1000)
                page.screenshot(path=f"test_outputs/hcaptcha_debug_{ts}.png")
            except Exception:
                pass

        if clicked or submitted:
            logger.info(
                "[Browser] hCaptcha 图像题尝试: prompt=%s mode=%s mask=%s diff=%s clicked=%s submitted=%s submit=%s size=%sx%s grid=%s hits=%s",
                (prompt or info.get("bodyText", "")[:80]).strip(),
                mode,
                indices,
                diff_indices,
                clicked,
                submitted,
                (picked_submit or info.get("submitText") or ""),
                width,
                height,
                {
                    "x": int(grid_rect.get("x") or 0) if isinstance(grid_rect, dict) else 0,
                    "y": int(grid_rect.get("y") or 0) if isinstance(grid_rect, dict) else 0,
                    "w": int(grid_rect.get("w") or 0) if isinstance(grid_rect, dict) else 0,
                    "h": int(grid_rect.get("h") or 0) if isinstance(grid_rect, dict) else 0,
                },
                hits[:3],
            )
            try:
                hints = info.get("selectedHints") or []
                if hints:
                    logger.info("[Browser] hCaptcha selected hints: %s", hints[:8])
            except Exception:
                pass
            return True
        return False

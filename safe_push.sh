#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
用法:
  ./safe_push.sh [选项]

功能:
  1) 可选自动提交（检测到工作区变更时）
  2) 强制推送（默认 --force-with-lease，更安全）
  3) 推送后自动校验远端分支 HEAD

选项:
  -m, --message <msg>     自动提交使用的 commit message
      --no-commit         有变更时不自动提交（默认: 会要求提供 -m 才能自动提交）
      --remote <name>     远端名（默认: remote.pushDefault 或 origin）
      --branch <name>     分支名（默认: 当前分支）
      --force             使用 --force（默认: --force-with-lease）
      --dry-run           仅演练推送，不真正推送
      --allow-sensitive   允许提交本地敏感配置文件（默认禁止）
  -h, --help              显示帮助

示例:
  ./safe_push.sh -m "fix: retry confirm flow"
  ./safe_push.sh --remote origin --branch main -m "chore: sync"
  ./safe_push.sh --dry-run
USAGE
}

err() {
  echo "[safe-push] ❌ $*" >&2
  exit 1
}

note() {
  echo "[safe-push] $*"
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  err "当前目录不是 Git 仓库"
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

remote_default="$(git config --get remote.pushDefault || true)"
if [[ -z "${remote_default}" ]]; then
  remote_default="origin"
fi

branch_default="$(git rev-parse --abbrev-ref HEAD)"

REMOTE="$remote_default"
BRANCH="$branch_default"
COMMIT_MSG=""
NO_COMMIT=0
FORCE_MODE="--force-with-lease"
DRY_RUN=0
ALLOW_SENSITIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      [[ $# -ge 2 ]] || err "参数 $1 需要一个值"
      COMMIT_MSG="$2"
      shift 2
      ;;
    --no-commit)
      NO_COMMIT=1
      shift
      ;;
    --remote)
      [[ $# -ge 2 ]] || err "参数 $1 需要一个值"
      REMOTE="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || err "参数 $1 需要一个值"
      BRANCH="$2"
      shift 2
      ;;
    --force)
      FORCE_MODE="--force"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-sensitive)
      ALLOW_SENSITIVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "未知参数: $1（用 -h 查看帮助）"
      ;;
  esac
done

# 基础校验
if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  err "远端不存在: $REMOTE"
fi

push_url="$(git remote get-url --push "$REMOTE" 2>/dev/null || true)"
if [[ -z "$push_url" || "$push_url" == "NO_PUSH" ]]; then
  err "远端 $REMOTE 的 push URL 不可用（当前: ${push_url:-<empty>}）"
fi

# 远端同步（分支不存在时忽略）
if ! git fetch "$REMOTE" "$BRANCH" --prune >/dev/null 2>&1; then
  note "提示: 远端分支 ${REMOTE}/${BRANCH} 可能尚不存在，继续执行"
fi

status_porcelain="$(git status --porcelain)"
if [[ -n "$status_porcelain" ]]; then
  if [[ "$NO_COMMIT" -eq 1 ]]; then
    err "检测到工作区变更，且设置了 --no-commit；请先手动提交"
  fi

  [[ -n "$COMMIT_MSG" ]] || err "检测到工作区变更，请提供 -m/--message 自动提交"

  note "检测到变更，准备自动提交"
  git add -A

  if [[ "$ALLOW_SENSITIVE" -ne 1 ]]; then
    sensitive_regex='^(config\.noproxy\.json|config\.noproxy\.card2\.json|config\.card2\.noproxy\.json|config\.json\.bak_before_card_test)$'
    staged_files="$(git diff --cached --name-only || true)"
    if [[ -n "$staged_files" ]] && echo "$staged_files" | grep -E "$sensitive_regex" >/dev/null 2>&1; then
      echo "[safe-push] 以下敏感文件被暂存，默认禁止提交:" >&2
      echo "$staged_files" | grep -E "$sensitive_regex" >&2 || true
      err "如确需提交，追加参数 --allow-sensitive"
    fi
  fi

  if git diff --cached --quiet; then
    note "没有可提交内容（可能都是 ignored 文件）"
  else
    git commit -m "$COMMIT_MSG"
  fi
else
  note "工作区干净，无需新提交"
fi

local_head="$(git rev-parse "$BRANCH")"
note "当前分支: $BRANCH"
note "当前提交: ${local_head:0:12}"
note "推送目标: $REMOTE ($push_url)"
note "推送模式: $FORCE_MODE"

push_cmd=(git push "$FORCE_MODE")
if [[ "$DRY_RUN" -eq 1 ]]; then
  push_cmd+=(--dry-run)
fi
push_cmd+=("$REMOTE" "$BRANCH")

note "执行: ${push_cmd[*]}"
"${push_cmd[@]}"

# 推送后校验
remote_head="$(git ls-remote "$REMOTE" "refs/heads/$BRANCH" | awk '{print $1}')"
if [[ -z "$remote_head" ]]; then
  note "⚠️ 未读取到远端 ${REMOTE}/${BRANCH} 的 HEAD（可能是权限/网络问题）"
  exit 0
fi

if [[ "$remote_head" == "$local_head" ]]; then
  note "✅ 完成：远端 ${REMOTE}/${BRANCH} = ${remote_head:0:12}"
else
  note "⚠️ 校验不一致：local=${local_head:0:12}, remote=${remote_head:0:12}"
  exit 1
fi

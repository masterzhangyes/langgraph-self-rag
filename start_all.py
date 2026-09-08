# -*- coding: utf-8 -*-
"""
智能问答系统 - 一键启动脚本
=====================================
- 自动检查并启动 后端(FastAPI :8000) 与 前端(Vite :5173)
- 依赖缺失时自动安装（首次安装较慢，请耐心等待）
- 服务已在运行时自动跳过，可放心重复执行

用法:
    python start_all.py              启动前后端（已运行则跳过）
    python start_all.py --quick      快速返回，服务在后台继续启动
    python start_all.py --no-browser 启动但不自动打开浏览器
    python start_all.py --status     查看当前运行状态
    python start_all.py --stop       停止监听 8000 / 5173 的服务
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
LOG_DIR = ROOT / "logs"
BACKEND_LOG = LOG_DIR / "uvicorn.out.log"
FRONTEND_LOG = LOG_DIR / "vite.out.log"
VENV_DIR = BACKEND / ".venv"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
REQ_FILE = BACKEND / "requirements.txt"

HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
BACKEND_HEALTH = BACKEND_URL + "/api/health"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"

# 国内镜像源（可自行修改/留空走官方源）
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
NPM_REGISTRY = "--registry=https://registry.npmmirror.com"

# ---------------------------------------------------------------- 通用工具

def _try_run(cmd, **kw):
    """运行命令并捕获输出（容错编码）。"""
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def _port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.8)
        return s.connect_ex((host, port)) == 0


def _http_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def _wait_http(url, timeout=90, step=1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _http_ok(url):
            return True
        time.sleep(step)
    return False


def _ready_timeout():
    """就绪等待上限：--quick 快速返回(服务仍在后台继续启动)，
    也可用环境变量 QA_START_TIMEOUT 控制，默认最长 90 秒。"""
    if "--quick" in sys.argv:
        return 8
    v = os.environ.get("QA_START_TIMEOUT")
    if v and v.isdigit():
        return min(int(v), 180)
    return 90


_SPAWN_FLAGS = (getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _spawn(cmd, cwd, logfile, env=None):
    """后台分离启动进程，输出写入日志文件，返回 PID。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    with open(logfile, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f,
                             stderr=subprocess.STDOUT,
                             creationflags=_SPAWN_FLAGS, env=e)
    return p.pid

# ---------------------------------------------------------------- 后端

def _base_python_candidates():
    """创建虚拟环境时的候补解释器（优先 3.10-3.12，避免过高版本不兼容）。"""
    cands = []
    # 1) 常见工具链自带解释器目录
    root = Path.home() / ".workbuddy" / "binaries" / "python" / "versions"
    if root.is_dir():
        for d in sorted(root.iterdir(), reverse=True):
            p = d / "python.exe"
            if p.exists() and d.name[:2] == "3.":
                cands.append(p)
    # 2) 当前解释器 / PATH 上的 python
    if sys.executable:
        cands.append(Path(sys.executable))
    seen, out = set(), []
    for p in cands:
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def _import_ok(py, mods):
    r = _try_run([str(py), "-c", "import " + ", ".join(mods)])
    return r.returncode == 0


def _pip_install(py, req):
    """安装 Python 依赖，优先镜像源，失败回退官方源。"""
    for desc, extra in (("镜像源", ["-i", PYPI_MIRROR]), ("官方源", [])):
        print(f"[后端] 正在用{desc}安装 Python 依赖 ...")
        r = subprocess.run([str(py), "-m", "pip", "install", "-r", str(req),
                            *extra, "--disable-pip-version-check"])
        if r.returncode == 0:
            return
    print("[后端] Python 依赖安装失败，请手动执行: "
          f"pip install -r \"{req}\" 查看详细报错")
    sys.exit(1)


def ensure_backend():
    """确保后端虚拟环境与依赖就绪，返回后端 Python 解释器路径。"""
    if VENV_PY.exists():
        print(f"[后端] 使用虚拟环境: {VENV_PY}")
        if not _import_ok(VENV_PY, ("fastapi", "uvicorn", "langchain")):
            print("[后端] 检测到依赖缺失，准备补齐 ...")
            _pip_install(VENV_PY, REQ_FILE)
        return VENV_PY

    print("[后端] 未发现虚拟环境 .venv，正在创建（约 1-3 分钟，请耐心）...")
    base = _base_python_candidates()[0] if _base_python_candidates() else None
    if base is None:
        print("[错误] 找不到可用的 Python，请安装 Python 3.10-3.12 后重试")
        sys.exit(1)
    subprocess.run([str(base), "-m", "venv", str(VENV_DIR)])
    if not VENV_PY.exists():
        print("[错误] 虚拟环境创建失败，请手动执行: python -m venv backend\\.venv")
        sys.exit(1)
    _pip_install(VENV_PY, REQ_FILE)
    return VENV_PY


def start_backend():
    if _http_ok(BACKEND_HEALTH, timeout=2):
        print(f"[后端] 已在运行: {BACKEND_HEALTH}")
        return
    py = ensure_backend()
    env = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1"}
    pid = _spawn([str(py), "-m", "uvicorn", "main:app",
                  "--host", HOST, "--port", str(BACKEND_PORT)],
                 BACKEND, BACKEND_LOG, env)
    print(f"[后端] 正在启动 (PID {pid})，等待就绪 ...")
    if _wait_http(BACKEND_HEALTH, timeout=_ready_timeout()):
        print(f"[后端] 就绪: {BACKEND_URL}    API 文档: {BACKEND_URL}/docs")
    else:
        print(f"[后端] 启动超时，请查看日志: {BACKEND_LOG}")

# ---------------------------------------------------------------- 前端

def _npm_cmd():
    return "npm.cmd" if os.name == "nt" else "npm"


def _has_vite():
    return (FRONTEND / "node_modules" / ".bin" / "vite.cmd").exists()


def ensure_frontend():
    """确保 vite 等前端依赖就绪。"""
    if _has_vite():
        return
    print("[前端] 缺少构建依赖 vite，开始 npm install（首次较慢）...")
    env = dict(os.environ)
    env.pop("NODE_ENV", None)  # 关键：否则 npm 会跳过 devDependencies
    r = subprocess.run([_npm_cmd(), "install", "--no-audit", "--no-fund",
                        NPM_REGISTRY], cwd=str(FRONTEND), env=env)
    if r.returncode != 0 or not _has_vite():
        print("[前端] npm install 失败，请手动进入 frontend 目录执行 npm install")
        sys.exit(1)


def start_frontend():
    if _port_open(FRONTEND_PORT):
        print(f"[前端] 已在运行: {FRONTEND_URL}")
        return
    ensure_frontend()
    env = {"NODE_ENV": "development"}
    if os.name == "nt":
        cmd = ["cmd", "/c",
               "npm run dev -- --host 0.0.0.0 --port " + str(FRONTEND_PORT)]
    else:
        cmd = ["npm", "run", "dev", "--", "--host", HOST,
               "--port", str(FRONTEND_PORT)]
    pid = _spawn(cmd, FRONTEND, FRONTEND_LOG, env)
    print(f"[前端] 正在启动 (PID {pid})，等待就绪 ...")
    if _wait_http(FRONTEND_URL, timeout=_ready_timeout()):
        print(f"[前端] 就绪: {FRONTEND_URL}")
    else:
        print(f"[前端] 启动超时，请查看日志: {FRONTEND_LOG}")

# ---------------------------------------------------------------- 其它命令

def print_status():
    b = _http_ok(BACKEND_HEALTH, timeout=2)
    f = _port_open(FRONTEND_PORT)
    print("后端 " + (f"{BACKEND_HEALTH}  ✔ 运行中" if b
                     else "✘ 未运行"))
    print("前端 " + (f"{FRONTEND_URL}      ✔ 运行中" if f
                     else "✘ 未运行"))
    print("日志目录:", LOG_DIR)


def stop_all():
    """停止占用 8000/5173 的进程树（覆盖 uvicorn --reload 的父 watcher）。"""
    ps = (
        "$ports = %d,%d\n"
        "$conns = Get-NetTCPConnection -State Listen -LocalPort $ports -ErrorAction SilentlyContinue\n"
        "$targets = @($conns | Select-Object -ExpandProperty OwningProcess -Unique)\n"
        "if ($targets.Count -eq 0) { Write-Output 'NO_LISTENER'; exit }\n"
        "$procs = Get-CimInstance Win32_Process\n"
        "$done = @{}\n"
        "foreach ($t in $targets) {\n"
        "  if ($done[$t]) { continue }\n"
        "  $cur = $t; $root = $t; $depth = 0\n"
        "  while ($cur -and $depth -lt 12) {\n"
        "    $p = $procs | Where-Object { $_.ProcessId -eq $cur } | Select-Object -First 1\n"
        "    if (-not $p) { break }\n"
        "    $name = $p.Name.ToLower()\n"
        "    $root = $p.ProcessId; $done[$t] = $true\n"
        "    $parent = $p.ParentProcessId\n"
        "    if ($parent -le 4 -or $name -notmatch '^(cmd|python|node|npm|conhost|powershell)') { break }\n"
        "    $cur = $parent; $depth++\n"
        "  }\n"
        "  taskkill /PID $root /T /F 2>$null | Out-Null\n"
        "}\n"
        "Write-Output 'DONE'\n"
    ) % (BACKEND_PORT, FRONTEND_PORT)
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if "NO_LISTENER" in r.stdout:
        print("当前没有监听 %d / %d 的服务" % (BACKEND_PORT, FRONTEND_PORT))
    else:
        print("已停止监听 %d / %d 端口的服务（含相关进程树）"
              % (BACKEND_PORT, FRONTEND_PORT))

# ---------------------------------------------------------------- 主流程

def main():
    args = sys.argv[1:]

    if "--stop" in args:
        stop_all()
        return
    if "--status" in args:
        print_status()
        return

    banner = "=" * 56
    print(banner)
    print("   智能问答系统 v2.1  -  一键启动")
    print(banner)
    print("  [1/2] 后端  FastAPI   : http://localhost:%d" % BACKEND_PORT)
    print("  [2/2] 前端  React+Vite: http://localhost:%d" % FRONTEND_PORT)
    print(banner)
    print()

    start_backend()
    print()
    start_frontend()

    print()
    print("=" * 56)
    ok_b = _http_ok(BACKEND_HEALTH, timeout=2)
    ok_f = _port_open(FRONTEND_PORT)
    if ok_b and ok_f:
        print("  启动完成！访问: http://localhost:%d" % FRONTEND_PORT)
    elif ok_b:
        print("  后端已就绪，前端启动超时，见日志: %s" % FRONTEND_LOG)
    elif ok_f:
        print("  前端已就绪，后端启动超时，见日志: %s" % BACKEND_LOG)
    else:
        print("  启动异常，请查看日志目录: %s" % LOG_DIR)
    print("  停止服务: python start_all.py --stop")
    print("  状态查看: python start_all.py --status")
    print("=" * 56)

    if "--no-browser" not in args:
        try:
            webbrowser.open(FRONTEND_URL)
        except Exception:
            pass


if __name__ == "__main__":
    main()

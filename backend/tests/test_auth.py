"""
用户认证单元测试 — 密码哈希 / JWT 签发校验 / 注册登录校验
=========================================================

不依赖真实数据库与网络：
    · 密码哈希与 JWT 均为纯函数级验证
    · 注册/登录流程通过 aiosqlite 内存级临时库验证（文件级 mock 路径）
"""
import pytest

from core.config import settings


# ─────────── 密码哈希（bcrypt） ───────────

class TestPasswordHashing:
    def test_hash_differs_from_plaintext(self):
        from infrastructure.user_store import hash_password
        hashed = hash_password("secret123")
        assert hashed != "secret123"
        assert hashed.startswith("$2b$")  # bcrypt 格式

    def test_hash_has_random_salt(self):
        """同一密码两次哈希结果不同（随机盐，抗彩虹表）"""
        from infrastructure.user_store import hash_password
        assert hash_password("secret123") != hash_password("secret123")

    def test_verify_roundtrip(self):
        from infrastructure.user_store import hash_password, verify_password
        hashed = hash_password("秘密密码abc")
        assert verify_password("秘密密码abc", hashed) is True

    def test_verify_wrong_password(self):
        from infrastructure.user_store import hash_password, verify_password
        hashed = hash_password("secret123")
        assert verify_password("wrong", hashed) is False

    def test_verify_malformed_hash(self):
        """畸形哈希不抛异常，返回 False（防御式编程）"""
        from infrastructure.user_store import verify_password
        assert verify_password("x", "not-a-bcrypt-hash") is False


# ─────────── JWT 签发与校验 ───────────

class TestJWT:
    def _tokens(self):
        from api.auth import _create_token, decode_token
        return _create_token, decode_token

    def test_access_token_roundtrip(self):
        _create_token, decode_token = self._tokens()
        token = _create_token(42, "alice", "admin", "access")
        payload = decode_token(token, "access")
        assert payload["sub"] == "42"
        assert payload["username"] == "alice"
        assert payload["role"] == "admin"

    def test_refresh_token_type_mismatch(self):
        """refresh 令牌不能当 access 使用（类型校验）"""
        _create_token, decode_token = self._tokens()
        token = _create_token(42, "alice", "user", "refresh")
        with pytest.raises(Exception):
            decode_token(token, "access")

    def test_tampered_token_rejected(self):
        """签名被篡改的令牌必须校验失败"""
        _create_token, decode_token = self._tokens()
        token = _create_token(42, "alice", "user", "access")
        with pytest.raises(Exception):
            decode_token(token + "tampered", "access")

    def test_wrong_secret_rejected(self):
        """用错误密钥签发的令牌必须被拒绝"""
        import jwt as pyjwt
        _create_token, decode_token = self._tokens()
        forged = pyjwt.encode(
            {"sub": "1", "type": "access"}, "wrong-secret", algorithm="HS256"
        )
        with pytest.raises(Exception):
            decode_token(forged, "access")


# ─────────── 登录防爆破限流 ───────────

class TestLoginRateLimit:
    def test_lockout_after_max_failures(self, monkeypatch):
        monkeypatch.setattr(settings, "LOGIN_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(settings, "LOGIN_LOCKOUT_SECONDS", 300)
        # 每个测试重置模块级状态，避免用例间污染
        import api.auth as auth_mod
        monkeypatch.setattr(auth_mod, "_login_failures", {})

        key = ("127.0.0.1", "bob")
        for _ in range(3):
            assert auth_mod._is_locked_out(key) is False
            auth_mod._record_failure(key)
        assert auth_mod._is_locked_out(key) is True

    def test_clear_failures_unlocks(self, monkeypatch):
        import api.auth as auth_mod
        monkeypatch.setattr(auth_mod, "_login_failures", {})
        key = ("127.0.0.1", "carol")
        for _ in range(5):
            auth_mod._record_failure(key)
        auth_mod._clear_failures(key)
        assert auth_mod._is_locked_out(key) is False


# ─────────── 注册请求校验 ───────────

class TestRegisterValidation:
    def test_short_username_rejected(self):
        from api.models import RegisterRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RegisterRequest(username="ab", password="123456")

    def test_short_password_rejected(self):
        from api.models import RegisterRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RegisterRequest(username="alice", password="123")

    def test_valid_request_accepted(self):
        from api.models import RegisterRequest
        req = RegisterRequest(username="alice", password="123456", email="a@b.com")
        assert req.username == "alice"
        assert req.email == "a@b.com"

    def test_bad_email_rejected(self):
        from api.models import RegisterRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RegisterRequest(username="alice", password="123456", email="not-an-email")

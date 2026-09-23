"""
知识库用户隔离测试（v2.3）
==========================

验证「按用户隔离的知识库」的权限边界:
    · 私有库仅本人（及管理员）可见 / 可写 / 可检索
    · 公共库所有人可读，仅管理员可写
    · 他人私有库访问返回 404（不泄露存在性）
    · 未登录写操作返回 401

通过 FastAPI TestClient 走完整 HTTP 栈，
向量索引函数（create/clear_knowledge_base）打桩避免真实 Embedding 调用。
"""
import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """独立临时环境的 TestClient（数据库 + 向量目录均为临时路径）"""
    import os

    monkeypatch.setattr("core.config.settings.DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setattr("core.config.settings.VECTOR_DB_PATH", str(tmp_path / "chroma_db" / "v.db"))
    monkeypatch.setattr("core.config.settings.LOG_PATH", str(tmp_path / "app.log"))

    # 模拟遗留部署: 磁盘上已有 chroma_default 目录（非空），
    # 启动迁移会把它注册为公共知识库 → 供公共库写保护用例使用
    legacy_dir = tmp_path / "chroma_db" / "chroma_default"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "dummy.bin").write_bytes(b"x")

    # 上传目录指向临时目录，避免污染仓库
    import main as main_module
    monkeypatch.setattr(main_module, "UPLOAD_DIR", str(tmp_path / "uploads"))
    os.makedirs(tmp_path / "uploads", exist_ok=True)

    with TestClient(main_module.app) as c:
        yield c


@pytest.fixture()
def users(client):
    """admin（首个注册自动成为管理员）+ 两个普通用户，返回 (admin, alice, bob) 的 token"""
    def _register(username):
        r = client.post("/api/auth/register",
                        json={"username": username, "password": "pass123456"})
        assert r.status_code == 201, r.text
        return r.json()["access_token"]

    return _register("admin_user"), _register("alice"), _register("bob")


def _auth(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def _upload(client, token, kb_name):
    """上传一个 txt 文件到指定知识库（索引函数已打桩）"""
    files = {"files": ("doc.txt", io.BytesIO("测试文档内容".encode()), "text/plain")}
    return client.post(f"/api/kb/upload?kb_name={kb_name}",
                       files=files, headers=_auth(token))


@pytest.fixture(autouse=True)
def stub_vector_ops(monkeypatch):
    """打桩向量索引与清空，避免真实 Embedding 网络调用"""
    from core import rag_chain

    monkeypatch.setattr(rag_chain, "create_knowledge_base", lambda **kwargs: 5)
    monkeypatch.setattr(rag_chain, "clear_knowledge_base", lambda kb_name="default": None)


# ─────────── 注册表与物理名隔离 ───────────

class TestPhysicalIsolation:
    def test_private_kb_physical_name_scoped_by_user(self, client, users):
        """同名知识库在不同用户下映射到不同物理目录"""
        import asyncio
        _, alice, bob = users
        assert _upload(client, alice, "docs").status_code == 200
        assert _upload(client, bob, "docs").status_code == 200

        from infrastructure import kb_store
        a = asyncio.run(kb_store.get_kb("docs", 2))   # alice 的 user id
        b = asyncio.run(kb_store.get_kb("docs", 3))   # bob 的 user id
        assert a and b
        assert a["physical_name"] != b["physical_name"]
        assert a["physical_name"].startswith("u2__")
        assert b["physical_name"].startswith("u3__")

    def test_same_name_allowed_across_users(self, client, users):
        """不同用户可以创建同名私有知识库（逻辑名按用户命名空间隔离）"""
        _, alice, bob = users
        assert _upload(client, alice, "notes").status_code == 200
        assert _upload(client, bob, "notes").status_code == 200


# ─────────── 列表可见性 ───────────

class TestListVisibility:
    def test_user_sees_own_and_public_only(self, client, users):
        """普通用户: 本人私有库 + 公共库；看不到他人私有库"""
        _, alice, bob = users
        _upload(client, alice, "alice-secret")
        _upload(client, bob, "bob-secret")

        r = client.get("/api/kb/list", headers=_auth(alice))
        assert r.status_code == 200
        names = [k["name"] for k in r.json()["knowledge_bases"]]
        assert "alice-secret" in names
        assert "bob-secret" not in names
        assert "default" in names  # 公共兜底

    def test_admin_sees_all(self, client, users):
        """管理员: 可见所有用户的私有库，且带所有者用户名"""
        admin, alice, bob = users
        _upload(client, alice, "alice-secret")
        _upload(client, bob, "bob-secret")

        r = client.get("/api/kb/list", headers=_auth(admin))
        names = [k["name"] for k in r.json()["knowledge_bases"]]
        assert "alice-secret" in names and "bob-secret" in names

        by_name = {k["name"]: k for k in r.json()["knowledge_bases"]}
        assert by_name["alice-secret"]["owner"] == "alice"
        assert by_name["alice-secret"]["is_public"] is False

    def test_anonymous_sees_public_only(self, client, users):
        """未登录: 仅公共库"""
        _, alice, _ = users
        _upload(client, alice, "alice-secret")

        r = client.get("/api/kb/list")
        names = [k["name"] for k in r.json()["knowledge_bases"]]
        assert "alice-secret" not in names
        assert "default" in names


# ─────────── 越权防护 ───────────

class TestAccessControl:
    def test_other_user_stats_404(self, client, users):
        """他人私有库统计 → 404（不泄露存在性）"""
        _, alice, bob = users
        _upload(client, alice, "alice-secret")
        r = client.get("/api/kb/alice-secret/stats", headers=_auth(bob))
        assert r.status_code == 404

    def test_other_user_clear_404(self, client, users):
        """他人私有库清空 → 404"""
        _, alice, bob = users
        _upload(client, alice, "alice-secret")
        r = client.delete("/api/kb/alice-secret/clear", headers=_auth(bob))
        assert r.status_code == 404

    def test_admin_can_clear_others(self, client, users):
        """管理员可清空他人私有库"""
        admin, alice, _ = users
        _upload(client, alice, "alice-secret")
        r = client.delete("/api/kb/alice-secret/clear", headers=_auth(admin))
        assert r.status_code == 200

    def test_anonymous_upload_401(self, client, users):
        """未登录上传 → 401"""
        r = _upload(client, None, "default")
        assert r.status_code == 401

    def test_anonymous_chat_with_private_kb_404(self, client, users):
        """未登录检索他人私有库 → 404；default 公共兜底仍可用"""
        _, alice, _ = users
        _upload(client, alice, "alice-secret")

        r = client.post("/api/chat",
                        json={"query": "hi", "kb_names": ["alice-secret"]})
        assert r.status_code == 404

    def test_owner_chat_resolves_private_kb(self, client, users, monkeypatch):
        """本人检索自己的私有库 → 通过权限校验（LLM 调用打桩）"""
        import asyncio
        _, alice, _ = users
        _upload(client, alice, "alice-secret")

        captured = {}

        async def fake_chat(query, history=None, kb_name="default", session_id=None):
            captured["kb_name"] = kb_name
            return {"answer": "ok", "kb_status": "empty", "sources": []}

        from core import rag_chain
        monkeypatch.setattr(rag_chain, "chat", fake_chat)

        r = client.post("/api/chat",
                        json={"query": "hi", "kb_names": ["alice-secret"]},
                        headers=_auth(alice))
        assert r.status_code == 200
        assert captured["kb_name"].startswith("u2__")  # 解析为 alice 的物理目录


# ─────────── 公共库写保护 ───────────

class TestPublicWriteProtection:
    def test_normal_user_cannot_write_public(self, client, users):
        """普通用户写公共库 → 403，且提示新建个人库"""
        _, alice, _ = users
        # 先由管理员注册一个公共 default（上传打桩后即注册）
        admin, _, _ = users
        _upload(client, admin, "default")

        r = _upload(client, alice, "default")
        assert r.status_code == 403
        assert "公共知识库" in r.json()["detail"]

    def test_new_kb_created_as_private(self, client, users):
        """普通用户上传到全新名称 → 自动创建为该用户的私有库"""
        import asyncio
        _, alice, _ = users
        r = _upload(client, alice, "my-own-kb")
        assert r.status_code == 200

        from infrastructure import kb_store
        kb = asyncio.run(kb_store.get_kb("my-own-kb", None))   # 公共域不存在
        assert kb is None

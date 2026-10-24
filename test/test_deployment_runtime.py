from __future__ import annotations

import asyncio
import json
import threading
from unittest import mock

from services import deployment_runtime as deployment_module


def test_rollback_restarts_background_workers_after_drain(monkeypatch):
    """排空中回滚为 active 后，应等待旧线程结束并重新启动后台服务。"""
    from api import app as app_module

    runtime = mock.Mock()
    runtime.is_active.return_value = True
    runtime.is_draining.side_effect = [False, True, False, False]
    shutdown = mock.Mock()
    shutdown.is_set.side_effect = [False, False, False, False, True]
    old = mock.Mock()
    old.stop_event = threading.Event()
    old.request_stop.side_effect = old.stop_event.set
    old.alive.side_effect = [True, False]
    new = mock.Mock()
    new.stop_event = threading.Event()
    starter = mock.Mock(side_effect=[old, new])
    monkeypatch.setattr(app_module, "deployment_runtime", runtime)
    monkeypatch.setattr(app_module, "_start_background_workers", starter)

    app_module._deployment_supervisor(shutdown)

    assert starter.call_count == 2
    assert all(call.kwargs == {"reload_shared_state": True} for call in starter.call_args_list)
    assert old.stop_event.is_set()
    ready = [call for call in runtime.mark_transition.call_args_list if call.kwargs["ready"]]
    assert len(ready) == 2
    new.join.assert_called_once_with(timeout=5)


def _runtime(monkeypatch, tmp_path, slot: str = "blue"):
    monkeypatch.setenv("CHATGPT2API_DEPLOY_SLOT", slot)
    runtime = deployment_module.DeploymentRuntime()
    runtime.active_path = tmp_path / "active-slot"
    runtime.draining_path = tmp_path / "draining-slot"
    return runtime


def test_managed_slot_is_standby_until_explicitly_activated(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)

    assert runtime.managed is True
    assert runtime.is_active() is False
    assert runtime.accepts_generation_requests() is False

    runtime.active_path.write_text("blue\n", encoding="utf-8")
    runtime.mark_transition(
        ready=True,
        background_running=True,
        background_quiesced=False,
    )
    assert runtime.accepts_generation_requests() is True

    runtime.draining_path.write_text("blue\n", encoding="utf-8")
    assert runtime.accepts_generation_requests() is False
    assert runtime.status()["draining"] is True


def test_generation_gate_returns_retryable_503_while_draining(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    runtime.active_path.write_text("blue\n", encoding="utf-8")
    runtime.draining_path.write_text("blue\n", encoding="utf-8")
    monkeypatch.setattr(deployment_module, "deployment_runtime", runtime)
    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = deployment_module.DeploymentGateMiddleware(downstream)
    asyncio.run(
        middleware(
            {"type": "http", "method": "POST", "path": "/v1/images/generations"},
            receive,
            send,
        )
    )

    assert downstream_called is False
    assert sent[0]["status"] == 503
    assert (b"retry-after", b"3") in sent[0]["headers"]
    payload = json.loads(sent[1]["body"])
    assert payload["error"]["code"] == "deployment_draining"


def test_read_only_request_is_not_blocked_while_draining(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    runtime.active_path.write_text("blue\n", encoding="utf-8")
    runtime.draining_path.write_text("blue\n", encoding="utf-8")
    monkeypatch.setattr(deployment_module, "deployment_runtime", runtime)
    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        return None

    asyncio.run(
        deployment_module.DeploymentGateMiddleware(downstream)(
            {"type": "http", "method": "GET", "path": "/healthz"},
            receive,
            send,
        )
    )
    assert downstream_called is True


def test_task_polling_is_retryable_during_state_handoff(monkeypatch, tmp_path):
    runtime = _runtime(monkeypatch, tmp_path)
    runtime.active_path.write_text("blue\n", encoding="utf-8")
    runtime.draining_path.write_text("blue\n", encoding="utf-8")
    monkeypatch.setattr(deployment_module, "deployment_runtime", runtime)
    sent = []

    async def downstream(scope, receive, send):
        raise AssertionError("任务状态切换期间不应读取可能变化的旧槽快照")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(
        deployment_module.DeploymentGateMiddleware(downstream)(
            {"type": "http", "method": "GET", "path": "/api/image-tasks"},
            receive,
            send,
        )
    )
    assert sent[0]["status"] == 503

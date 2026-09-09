from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Event, Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api import accounts, ai, image_tasks, system
from api.errors import install_exception_handlers
from api.request_body_guard import ImageRequestBodyGuardMiddleware
from api.support import resolve_web_asset, start_limited_account_watcher
from services.backup_service import backup_service
from services.config import config
from services.deployment_runtime import DeploymentGateMiddleware, deployment_runtime
from services.image_service import start_image_cleanup_scheduler
from services.runtime_watchdog import start_runtime_watchdog


@dataclass
class _BackgroundWorkers:
    stop_event: Event
    threads: list[Thread]

    def request_stop(self) -> None:
        if deployment_runtime.managed:
            from services.editable_file_task_service import editable_file_task_service

            editable_file_task_service.enter_standby()
        self.stop_event.set()
        backup_service.stop()

    def alive(self) -> bool:
        return any(thread.is_alive() for thread in self.threads)

    def join(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))


def _start_background_workers(*, reload_shared_state: bool) -> _BackgroundWorkers:
    if reload_shared_state:
        # 旧槽已经进入排空状态后才会执行，保证候选槽拿到最新共享状态。
        from services.account_service import account_service
        from services.editable_file_task_service import editable_file_task_service
        from services.image_task_service import image_task_service

        account_service.reload_from_storage()
        image_task_service.activate_from_shared_state()
        editable_file_task_service.activate_from_shared_state()

    stop_event = Event()
    threads = [
        start_limited_account_watcher(stop_event),
        start_image_cleanup_scheduler(stop_event),
    ]
    backup_service.start()
    config.cleanup_old_images()
    watchdog_thread = start_runtime_watchdog(stop_event)
    if watchdog_thread is not None:
        threads.append(watchdog_thread)
    return _BackgroundWorkers(stop_event=stop_event, threads=threads)


def _deployment_supervisor(shutdown_event: Event) -> None:
    """保证只有 active 且非排空槽运行会写共享状态的后台服务。"""

    workers: _BackgroundWorkers | None = None
    deployment_runtime.mark_transition(
        ready=False,
        background_running=False,
        background_quiesced=True,
    )
    while not shutdown_event.is_set():
        should_run = (
            deployment_runtime.is_active() and not deployment_runtime.is_draining()
        )
        if should_run and workers is None:
            deployment_runtime.mark_transition(
                ready=False,
                background_running=False,
                background_quiesced=True,
            )
            try:
                workers = _start_background_workers(reload_shared_state=True)
            except Exception as exc:  # noqa: BLE001 - 激活失败必须留在待机槽并暴露状态
                deployment_runtime.mark_transition(
                    ready=False,
                    background_running=False,
                    background_quiesced=True,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            else:
                deployment_runtime.mark_transition(
                    ready=True,
                    background_running=True,
                    background_quiesced=False,
                )
        elif not should_run and workers is not None:
            deployment_runtime.mark_transition(
                ready=False,
                background_running=False,
                background_quiesced=False,
            )
            workers.request_stop()
            if not workers.alive():
                workers = None
                deployment_runtime.mark_transition(
                    ready=False,
                    background_running=False,
                    background_quiesced=True,
                )
        shutdown_event.wait(0.25)

    if workers is not None:
        workers.request_stop()
        workers.join(timeout=5)
    deployment_runtime.mark_transition(
        ready=False,
        background_running=False,
        background_quiesced=True,
    )


def create_app() -> FastAPI:
    app_version = config.app_version

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if deployment_runtime.managed:
            shutdown_event = Event()
            supervisor = Thread(
                target=_deployment_supervisor,
                args=(shutdown_event,),
                name="deployment-slot-supervisor",
                daemon=True,
            )
            supervisor.start()
            try:
                yield
            finally:
                shutdown_event.set()
                supervisor.join(timeout=7)
        else:
            workers = _start_background_workers(reload_shared_state=False)
            try:
                yield
            finally:
                workers.request_stop()
                workers.join(timeout=5)

    app = FastAPI(title="chatgpt2api", version=app_version, lifespan=lifespan)
    install_exception_handlers(app)
    app.add_middleware(ImageRequestBodyGuardMiddleware)
    app.add_middleware(DeploymentGateMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ai.create_router())
    app.include_router(accounts.create_router())
    app.include_router(image_tasks.create_router())
    app.include_router(system.create_router(app_version))

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_web(full_path: str):
        asset = resolve_web_asset(full_path)
        if asset is not None:
            return FileResponse(asset)
        if full_path.strip("/").startswith("_next/"):
            raise HTTPException(status_code=404, detail="Not Found")
        fallback = resolve_web_asset("")
        if fallback is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(fallback)

    return app

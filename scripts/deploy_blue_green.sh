#!/usr/bin/env bash
set -Eeuo pipefail

# chatgpt2api 蓝绿发布：候选槽待机启动，旧槽拒绝新生成任务并排空，候选槽
# 重载共享状态后再由 nginx graceful reload 原子切流。任何验收失败都会回滚。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${CHATGPT2API_BLUE_GREEN_COMPOSE_FILE:-docker-compose.blue-green.yml}"
CONTAINER_PREFIX="${CHATGPT2API_CONTAINER_PREFIX:-chatgpt2api}"
GATEWAY_CONTAINER="${CONTAINER_PREFIX}-gateway"
BLUE_CONTAINER="${CONTAINER_PREFIX}-blue"
GREEN_CONTAINER="${CONTAINER_PREFIX}-green"
LEGACY_CONTAINER="${CHATGPT2API_LEGACY_CONTAINER:-chatgpt2api-local}"
NGINX_DIR="$ROOT_DIR/deploy/nginx"
ACTIVE_CONFIG="$NGINX_DIR/active.conf"
STATE_DIR="$ROOT_DIR/data/deploy"
ACTIVE_STATE="$STATE_DIR/active-slot"
DRAIN_STATE="$STATE_DIR/draining-slot"
LOCK_FILE="$STATE_DIR/deploy.lock"
DRAIN_TIMEOUT_SECS="${CHATGPT2API_DRAIN_TIMEOUT_SECS:-360}"
HOST_PORT="${CHATGPT2API_HOST_PORT:-3000}"
REQUIRE_COS_VALIDATION="${CHATGPT2API_REQUIRE_COS_VALIDATION:-true}"
GATEWAY_IMAGE="${CHATGPT2API_GATEWAY_IMAGE:-docker.m.daocloud.io/library/nginx:1.27-alpine}"

mkdir -p "$NGINX_DIR" "$STATE_DIR"
touch "$LOCK_FILE"
chmod 600 "$LOCK_FILE"
LOCK_DIR="${LOCK_FILE}.d"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "已有 chatgpt2api 发布任务正在运行，本次发布已安全退出。" >&2
    exit 75
  fi
else
  # macOS 本地演练没有 flock，使用原子 mkdir；生产 Linux 仍使用内核文件锁。
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "已有 chatgpt2api 发布任务正在运行，本次发布已安全退出。" >&2
    exit 75
  fi
  cleanup_portable_lock() {
    local exit_code=$?
    rmdir "$LOCK_DIR" 2>/dev/null || true
    return "$exit_code"
  }
  trap cleanup_portable_lock EXIT
fi

if [[ ! -f .env ]]; then
  echo "缺少生产 .env，拒绝发布。" >&2
  exit 1
fi
if [[ "$(stat -c %a .env 2>/dev/null || stat -f %Lp .env)" != "600" ]]; then
  echo "生产 .env 权限必须为 600，拒绝发布。" >&2
  exit 1
fi

export CHATGPT2API_RELEASE_SHA
CHATGPT2API_RELEASE_SHA="$(git rev-parse --short=12 HEAD)"

container_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true
}

container_health() {
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$1" 2>/dev/null || true
}

slot_container() {
  [[ "$1" == "blue" ]] && printf '%s\n' "$BLUE_CONTAINER" || printf '%s\n' "$GREEN_CONTAINER"
}

slot_service() {
  printf 'app-%s\n' "$1"
}

read_active_slot() {
  local value=""
  if [[ -f "$ACTIVE_CONFIG" ]]; then
    value="$(sed -nE 's/.*server app-(blue|green):80.*/\1/p' "$ACTIVE_CONFIG" | head -n 1)"
  fi
  if [[ -z "$value" && -f "$ACTIVE_STATE" ]]; then
    value="$(tr -d '[:space:]' < "$ACTIVE_STATE")"
  fi
  if [[ "$value" == "blue" || "$value" == "green" ]]; then
    printf '%s\n' "$value"
  fi
}

established_connections() {
  local container="$1"
  if ! container_running "$container"; then
    echo 0
    return
  fi
  docker exec "$container" uv run python -c '
from pathlib import Path
count = 0
for name in ("/proc/net/tcp", "/proc/net/tcp6"):
    path = Path(name)
    if not path.exists():
        continue
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) > 3 and parts[1].rsplit(":", 1)[-1].upper() == "0050" and parts[3] == "01":
            count += 1
print(count)
'
}

wait_healthy() {
  local container="$1"
  local deadline=$((SECONDS + 150))
  while (( SECONDS < deadline )); do
    if [[ "$(container_health "$container")" == "healthy" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "容器 ${container} 未在 150 秒内健康。" >&2
  docker logs --tail 100 "$container" >&2 || true
  return 1
}

deployment_state_matches() {
  local container="$1"
  local expression="$2"
  docker exec "$container" uv run python -c '
import json, sys, urllib.request
payload = json.load(urllib.request.urlopen("http://127.0.0.1/healthz", timeout=3))
deployment = payload.get("deployment") or {}
expected = json.loads(sys.argv[1])
raise SystemExit(0 if all(deployment.get(key) == value for key, value in expected.items()) else 1)
' "$expression"
}

runtime_is_idle() {
  local container="$1"
  docker exec "$container" uv run python -c '
import json, urllib.request
payload = json.load(urllib.request.urlopen("http://127.0.0.1/healthz", timeout=3))
busy = 0
for name in ("image_runtime", "editable_file_runtime"):
    state = payload.get(name) or {}
    for key in ("active_jobs", "queued_jobs", "validating_jobs"):
        value = state.get(key, 0)
        busy += max(0, int(value)) if isinstance(value, (int, float)) else 0
raise SystemExit(0 if busy == 0 else 1)
'
}

wait_old_slot_quiesced() {
  local container="$1"
  local managed="$2"
  local deadline=$((SECONDS + DRAIN_TIMEOUT_SECS))
  while (( SECONDS < deadline )); do
    if runtime_is_idle "$container"; then
      if [[ "$managed" != true ]] || deployment_state_matches "$container" '{"background_quiesced":true}'; then
        return 0
      fi
    fi
    sleep 2
  done
  echo "旧槽未在 ${DRAIN_TIMEOUT_SECS} 秒内排空；保持原流量并取消上线。" >&2
  return 1
}

wait_slot_ready() {
  local container="$1"
  local slot="$2"
  local deadline=$((SECONDS + 90))
  while (( SECONDS < deadline )); do
    if deployment_state_matches "$container" "{\"slot\":\"$slot\",\"active\":true,\"ready\":true,\"background_running\":true}"; then
      return 0
    fi
    sleep 1
  done
  echo "候选槽 ${slot} 激活失败。" >&2
  docker logs --tail 100 "$container" >&2 || true
  return 1
}

validate_application() {
  local container="$1"
  docker exec "$container" uv run python -c '
import json, os, sys, urllib.request
health = json.load(urllib.request.urlopen("http://127.0.0.1/healthz", timeout=5))
assert health.get("healthy") is True, health
req = urllib.request.Request(
    "http://127.0.0.1/api/settings",
    headers={"Authorization": "Bearer " + os.environ["CHATGPT2API_AUTH_KEY"]},
)
settings = json.load(urllib.request.urlopen(req, timeout=5))["config"]
storage = settings["image_storage"]
if sys.argv[1].lower() not in {"0", "false", "no", "off"}:
    assert storage.get("managed_by_env") is True
    assert storage.get("has_cos_secret_id") is True
    assert storage.get("has_cos_secret_key") is True
assert int(settings["image_task_runtime"]["total_timeout_secs"]) > 0
' "$REQUIRE_COS_VALIDATION"
}

validate_cos() {
  local container="$1"
  docker exec "$container" uv run python -c '
import json, os, urllib.request
req = urllib.request.Request(
    "http://127.0.0.1/api/image-storage/test",
    method="POST",
    headers={"Authorization": "Bearer " + os.environ["CHATGPT2API_AUTH_KEY"]},
)
payload = json.load(urllib.request.urlopen(req, timeout=30))
assert (payload.get("result") or {}).get("ok") is True, payload
'
}

validate_network_path() {
  local container="$1"
  local service="$2"
  local network
  network="$(docker inspect -f '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{end}}' "$container")"
  docker run --rm --network "$network" "$GATEWAY_IMAGE" \
    wget -q -T 5 -O /dev/null "http://${service}:80/healthz"
}

write_candidate_config() {
  local slot="$1"
  local destination="$2"
  printf 'upstream chatgpt2api_active {\n    server app-%s:80;\n    keepalive 32;\n}\n' "$slot" > "$destination"
}

validate_nginx_candidate() {
  local candidate="$1"
  local network="$2"
  local temp_dir
  temp_dir="$(mktemp -d "$STATE_DIR/nginx-check.XXXXXX")"
  cp "$NGINX_DIR/gateway.conf" "$temp_dir/gateway.conf"
  cp "$candidate" "$temp_dir/active.conf"
  if ! docker run --rm --network "$network" -v "$temp_dir:/etc/nginx/conf.d:ro" \
    "$GATEWAY_IMAGE" nginx -t >/dev/null; then
    rm -rf "$temp_dir"
    return 1
  fi
  rm -rf "$temp_dir"
}

active_slot="$(read_active_slot)"
legacy_migration=false
bootstrap=false
if [[ -z "$active_slot" ]]; then
  if container_running "$LEGACY_CONTAINER"; then
    legacy_migration=true
  else
    bootstrap=true
  fi
  target_slot="blue"
else
  target_slot="$([[ "$active_slot" == "blue" ]] && echo green || echo blue)"
fi

target_container="$(slot_container "$target_slot")"
target_service="$(slot_service "$target_slot")"

# 两槽发布不能覆盖仍在排空连接的旧槽；宁可拒绝，也不杀掉用户请求。
if container_running "$target_container"; then
  pending="$(established_connections "$target_container")"
  if (( pending > 0 )); then
    echo "闲置槽 ${target_slot} 仍有 ${pending} 条连接，拒绝覆盖；稍后重新执行上线。" >&2
    exit 75
  fi
fi

echo "构建候选版本：${target_slot}（${CHATGPT2API_RELEASE_SHA}）"
docker compose -f "$COMPOSE_FILE" config --quiet
docker compose -f "$COMPOSE_FILE" build "$target_service"
docker compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate "$target_service"
wait_healthy "$target_container"
validate_application "$target_container"
if [[ "$REQUIRE_COS_VALIDATION" != false && "$REQUIRE_COS_VALIDATION" != 0 ]]; then
  validate_cos "$target_container"
fi
validate_network_path "$target_container" "$target_service"

network="$(docker inspect -f '{{range $name, $_ := .NetworkSettings.Networks}}{{$name}}{{end}}' "$target_container")"
candidate="$(mktemp "$STATE_DIR/active.XXXXXX")"
write_candidate_config "$target_slot" "$candidate"
validate_nginx_candidate "$candidate" "$network"

previous_config=""
if [[ -f "$ACTIVE_CONFIG" ]]; then
  previous_config="$(mktemp "$STATE_DIR/previous-active.XXXXXX")"
  cp "$ACTIVE_CONFIG" "$previous_config"
fi

state_changed=false
traffic_switched=false
release_succeeded=false

rollback() {
  local exit_code=$?
  [[ "$release_succeeded" == true ]] && return 0
  set +e
  echo "发布失败，正在恢复原槽。" >&2
  if [[ "$legacy_migration" == true || "$bootstrap" == true ]]; then
    rm -f "$ACTIVE_STATE" "$DRAIN_STATE"
    if container_running "$GATEWAY_CONTAINER"; then
      docker stop -t 20 "$GATEWAY_CONTAINER" >/dev/null
    fi
    if [[ "$legacy_migration" == true ]]; then
      docker start "$LEGACY_CONTAINER" >/dev/null
    fi
  elif [[ -n "$active_slot" ]]; then
    printf '%s\n' "$active_slot" > "$ACTIVE_STATE"
    chmod 600 "$ACTIVE_STATE"
    rm -f "$DRAIN_STATE"
    old_container="$(slot_container "$active_slot")"
    wait_slot_ready "$old_container" "$active_slot" || true
    if [[ -n "$previous_config" && -f "$previous_config" ]]; then
      cp "$previous_config" "$ACTIVE_CONFIG"
      if container_running "$GATEWAY_CONTAINER"; then
        docker exec "$GATEWAY_CONTAINER" nginx -t >/dev/null 2>&1
        docker exec "$GATEWAY_CONTAINER" nginx -s reload >/dev/null 2>&1
      fi
    fi
  fi
  rm -f "$candidate" "$previous_config"
  exit "$exit_code"
}
trap rollback ERR INT TERM

if [[ "$bootstrap" != true ]]; then
  drain_slot="$([[ "$legacy_migration" == true ]] && echo legacy || echo "$active_slot")"
  printf '%s\n' "$drain_slot" > "$DRAIN_STATE"
  chmod 600 "$DRAIN_STATE"
  echo "旧槽进入排空状态：不再接收新生成任务，已进入的请求继续完成。"
fi
if [[ "$legacy_migration" == true ]]; then
  wait_old_slot_quiesced "$LEGACY_CONTAINER" false
  # 首次迁移的旧镜像不认识蓝绿状态，只能先优雅停止；此后发布均为无损切流。
  docker stop -t "$DRAIN_TIMEOUT_SECS" "$LEGACY_CONTAINER" >/dev/null
elif [[ "$bootstrap" != true ]]; then
  old_container="$(slot_container "$active_slot")"
  wait_old_slot_quiesced "$old_container" true
fi

printf '%s\n' "$target_slot" > "$ACTIVE_STATE"
chmod 600 "$ACTIVE_STATE"
state_changed=true
wait_slot_ready "$target_container" "$target_slot"
validate_application "$target_container"

mv "$candidate" "$ACTIVE_CONFIG"
chmod 644 "$ACTIVE_CONFIG"
if [[ "$legacy_migration" == true || "$bootstrap" == true ]] || ! container_running "$GATEWAY_CONTAINER"; then
  docker compose -f "$COMPOSE_FILE" up -d --no-deps gateway
else
  docker exec "$GATEWAY_CONTAINER" nginx -t
  docker exec "$GATEWAY_CONTAINER" nginx -s reload
fi
traffic_switched=true

wait_healthy "$GATEWAY_CONTAINER"
curl -fsS --max-time 10 "http://127.0.0.1:${HOST_PORT}/healthz" >/dev/null
rm -f "$DRAIN_STATE"
rm -f "$previous_config"
release_succeeded=true
trap - ERR INT TERM
echo "流量已切换到 ${target_slot} 槽。"

if [[ -n "$active_slot" ]]; then
  old_container="$(slot_container "$active_slot")"
  deadline=$((SECONDS + DRAIN_TIMEOUT_SECS))
  zero_checks=0
  while (( SECONDS < deadline )); do
    pending="$(established_connections "$old_container")"
    if (( pending == 0 )); then
      zero_checks=$((zero_checks + 1))
      if (( zero_checks >= 3 )); then
        docker stop -t 20 "$old_container" >/dev/null || true
        echo "旧槽 ${active_slot} 已排空并停止。"
        break
      fi
    else
      zero_checks=0
    fi
    sleep 2
  done
  if container_running "$old_container"; then
    pending="$(established_connections "$old_container")"
    echo "旧槽 ${active_slot} 仍有 ${pending} 条连接，继续保留运行；下次上线不会覆盖它。"
  fi
fi

echo "蓝绿发布完成：active=${target_slot} release=${CHATGPT2API_RELEASE_SHA}"

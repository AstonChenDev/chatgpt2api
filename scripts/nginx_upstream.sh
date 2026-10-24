#!/usr/bin/env bash

# 为蓝绿网关生成动态解析的上游配置。Docker 重建/重启应用时可能改变 IP，
# 必须通过 Docker DNS 定期刷新，不能把启动时的解析结果永久留在 worker 中。
write_candidate_config() {
  local slot="$1"
  local destination="$2"
  case "$slot" in
    blue|green) ;;
    *) echo "无效的应用槽：$slot" >&2; return 2 ;;
  esac
  printf 'upstream chatgpt2api_active {\n    zone chatgpt2api_active 64k;\n    resolver 127.0.0.11 valid=5s ipv6=off;\n    resolver_timeout 2s;\n    server app-%s:80 resolve;\n    keepalive 32;\n}\n' "$slot" > "$destination"
}

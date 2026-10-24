from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "scripts" / "nginx_upstream.sh"


class NginxUpstreamConfigTests(unittest.TestCase):
    def test_both_slots_use_docker_dns_and_shared_resolvable_upstream(self) -> None:
        for slot in ("blue", "green"):
            with self.subTest(slot=slot), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "active.conf"
                subprocess.run(
                    ["bash", "-c", 'source "$1"; write_candidate_config "$2" "$3"',
                     "renderer", str(RENDERER), slot, str(destination)],
                    check=True,
                )
                config = destination.read_text(encoding="utf-8")
                self.assertIn(f"server app-{slot}:80 resolve;", config)
                self.assertIn("resolver 127.0.0.11 valid=5s ipv6=off;", config)
                self.assertIn("zone chatgpt2api_active 64k;", config)
                self.assertIn("keepalive 32;", config)

    def test_invalid_slot_does_not_overwrite_existing_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "active.conf"
            destination.write_text("existing config\n", encoding="utf-8")
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; write_candidate_config "$2" "$3"',
                 "renderer", str(RENDERER), "invalid; injected", str(destination)],
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(destination.read_text(encoding="utf-8"), "existing config\n")

    @unittest.skipUnless(
        os.getenv("CHATGPT2API_TEST_DOCKER_DNS") == "1" and shutil.which("docker"),
        "设置 CHATGPT2API_TEST_DOCKER_DNS=1 后执行隔离的 Docker DNS 变更演练",
    )
    def test_gateway_recovers_after_backend_ip_changes_without_reload(self) -> None:
        """替换后端并占住旧 IP，验证网关无须 reload 就能连接新地址。"""
        image = "docker.m.daocloud.io/library/nginx:1.27-alpine"
        prefix = "chatgpt2api-dns-test-" + uuid.uuid4().hex[:10]
        network = prefix + "-network"
        before, gateway, blocker, after = [prefix + suffix for suffix in ("-before", "-gateway", "-blocker", "-after")]
        created: list[str] = []

        def docker(*args: str) -> str:
            result = subprocess.run(
                ["docker", *args], capture_output=True, text=True, timeout=40,
            )
            if result.returncode:
                raise RuntimeError(result.stderr)
            return result.stdout.strip()

        def wait_proxy() -> None:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                response = subprocess.run(
                    ["docker", "exec", gateway, "wget", "-q", "-T", "1", "-O", "/dev/null", "http://127.0.0.1/"],
                    capture_output=True, timeout=5,
                )
                if response.returncode == 0:
                    return
                time.sleep(0.5)
            self.fail("网关未在 DNS 刷新窗口内恢复")

        with tempfile.TemporaryDirectory() as directory:
            conf_dir = Path(directory)
            shutil.copyfile(ROOT / "deploy/nginx/gateway.conf", conf_dir / "gateway.conf")
            subprocess.run(
                ["bash", "-c", 'source "$1"; write_candidate_config blue "$2"',
                 "renderer", str(RENDERER), str(conf_dir / "active.conf")], check=True,
            )
            networks = json.loads(docker("network", "inspect", *docker("network", "ls", "-q").split()))
            occupied = [ipaddress.ip_network(config["Subnet"]) for existing in networks
                        for config in ((existing.get("IPAM") or {}).get("Config") or []) if config.get("Subnet")]
            subnet = next((ipaddress.ip_network(f"10.254.{octet}.0/24") for octet in range(1, 255)
                           if not any(ipaddress.ip_network(f"10.254.{octet}.0/24").overlaps(used)
                                      for used in occupied if used.version == 4)), None)
            if subnet is None:
                self.skipTest("没有空闲子网可用于隔离的 DNS 变更演练")
            docker("network", "create", "--subnet", str(subnet), network)
            try:
                created.append(before)
                docker("run", "-d", "--name", before, "--network", network, "--network-alias", "app-blue", image)
                created.append(gateway)
                docker("run", "-d", "--name", gateway, "--network", network,
                       "-v", f"{conf_dir}:/etc/nginx/conf.d:ro", image)
                wait_proxy()
                gateway_id = docker("inspect", "-f", "{{.Id}}", gateway)
                ip_format = "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"
                old_ip = docker("inspect", "-f", ip_format, before)
                docker("rm", "-f", before)
                created.append(blocker)
                docker("run", "-d", "--name", blocker, "--network", network, "--ip", old_ip, image, "sleep", "120")
                created.append(after)
                docker("run", "-d", "--name", after, "--network", network, "--network-alias", "app-blue", image)
                self.assertNotEqual(docker("inspect", "-f", ip_format, after), old_ip)
                wait_proxy()
                self.assertEqual(docker("inspect", "-f", "{{.Id}}", gateway), gateway_id)
            finally:
                if created:
                    subprocess.run(["docker", "rm", "-f", *created], capture_output=True, timeout=40)
                subprocess.run(["docker", "network", "rm", network], capture_output=True, timeout=40)


if __name__ == "__main__":
    unittest.main()

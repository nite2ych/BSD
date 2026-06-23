"""Deploy live_bsd artifacts to a V853 board over serial using HTTP pull.

The board shell is reached through a serial port. Files are served from a
temporary local HTTP server and pulled on the board with wget. This is much
faster than base64 transfer over 115200 baud serial.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import serial


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host-ip", default="192.168.144.100")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--board-dir", default="/mnt/UDISK")
    parser.add_argument("--nb", type=Path, required=True)
    parser.add_argument("--remote-nb", default="bsd_v7_yolo26n_640.nb")
    parser.add_argument("--live-bsd", type=Path, default=Path("artifacts/board_bin/live_bsd"))
    parser.add_argument("--test-npu-direct", type=Path, default=Path("artifacts/board_bin/test_npu_direct"))
    parser.add_argument("--init-script", type=Path, default=Path("deployment/board/init.d/bsd_live"))
    parser.add_argument("--install-autostart", action="store_true")
    parser.add_argument("--model-size", type=int, default=640)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--disp-conf", type=float, default=0.5)
    parser.add_argument("--person-conf", type=float, default=0.5)
    parser.add_argument("--run-mode", choices=["none", "headless", "preview"], default="none")
    parser.add_argument("--log-name", default="bsd_v7_live.log")
    return parser.parse_args()


def copy_payload(args: argparse.Namespace, serve_dir: Path) -> dict[str, Path]:
    payload = {
        args.remote_nb: args.nb,
        "live_bsd": args.live_bsd,
        "test_npu_direct": args.test_npu_direct,
    }
    if args.install_autostart:
        payload["bsd_live_init"] = args.init_script
    for remote_name, src in payload.items():
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, serve_dir / remote_name)
    return payload


def start_http(serve_dir: Path, port: int) -> socketserver.TCPServer:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    httpd = ReusableTCPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def read_available(ser: serial.Serial, idle_rounds: int = 4) -> str:
    data = bytearray()
    idle = 0
    while idle < idle_rounds:
        waiting = ser.in_waiting
        if waiting:
            data.extend(ser.read(waiting))
            idle = 0
        else:
            idle += 1
            time.sleep(0.25)
    return data.decode("utf-8", errors="replace")


def send_cmd(ser: serial.Serial, cmd: str, wait: float = 1.0) -> str:
    print(f"\n=== board$ {cmd} ===")
    ser.write((cmd + "\n").encode("utf-8"))
    time.sleep(wait)
    out = read_available(ser)
    if out:
        print(out)
    return out


def board_wget(ser: serial.Serial, board_dir: str, host_ip: str, port: int, name: str) -> None:
    url = f"http://{host_ip}:{port}/{name}"
    send_cmd(ser, f"cd {board_dir} && wget -O {name} {url}", wait=3.0)


def board_wget_to(ser: serial.Serial, host_ip: str, port: int, name: str, remote_path: str) -> None:
    url = f"http://{host_ip}:{port}/{name}"
    send_cmd(ser, f"wget -O {remote_path} {url}", wait=3.0)


def run_live_command(args: argparse.Namespace) -> str:
    return (
        f"{args.board_dir}/live_bsd {args.board_dir}/{args.remote_nb} "
        f"{args.det_conf} {args.nms} {args.disp_conf} {args.person_conf} "
        f"{args.cam_width} {args.cam_height} {args.run_mode} {args.model_size}"
    )


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="bsd_deploy_") as tmp:
        serve_dir = Path(tmp)
        copy_payload(args, serve_dir)
        httpd = start_http(serve_dir, args.port)
        print(f"Serving {serve_dir} at http://{args.host_ip}:{args.port}/")
        try:
            ser = serial.Serial(args.serial, args.baud, timeout=0.5)
            try:
                send_cmd(ser, "echo BOARD_READY", wait=0.8)
                send_cmd(ser, f"mkdir -p {args.board_dir}", wait=0.5)
                board_wget(ser, args.board_dir, args.host_ip, args.port, args.remote_nb)
                board_wget(ser, args.board_dir, args.host_ip, args.port, "live_bsd")
                board_wget(ser, args.board_dir, args.host_ip, args.port, "test_npu_direct")
                send_cmd(
                    ser,
                    f"chmod +x {args.board_dir}/live_bsd {args.board_dir}/test_npu_direct",
                    wait=0.5,
                )
                send_cmd(
                    ser,
                    f"ls -lh {args.board_dir}/{args.remote_nb} {args.board_dir}/live_bsd "
                    f"{args.board_dir}/test_npu_direct",
                    wait=0.8,
                )
                send_cmd(
                    ser,
                    f"printf magic=; dd if={args.board_dir}/{args.remote_nb} bs=1 count=4 2>/dev/null; echo",
                    wait=0.8,
                )
                if args.install_autostart:
                    board_wget_to(ser, args.host_ip, args.port, "bsd_live_init", "/etc/init.d/bsd_live")
                    send_cmd(
                        ser,
                        "chmod +x /etc/init.d/bsd_live && "
                        "/etc/init.d/bsd_live enable && "
                        "ln -sf bsd_live /etc/init.d/S99bsd_live",
                        wait=1.0,
                    )
                    send_cmd(
                        ser,
                        "ls -l /etc/init.d/bsd_live /etc/init.d/S99bsd_live /etc/rc.d/S99bsd_live",
                        wait=0.8,
                    )
                if args.run_mode != "none":
                    log_path = f"{args.board_dir}/{args.log_name}"
                    cmd = (
                        f"killall live_bsd 2>/dev/null; rm -f {log_path}; "
                        f"{run_live_command(args)} > {log_path} 2>&1 & echo LIVE_PID=$!"
                    )
                    send_cmd(ser, cmd, wait=1.0)
                    time.sleep(8)
                    send_cmd(ser, "ps | grep live_bsd", wait=0.8)
                    send_cmd(ser, f"cat {log_path}", wait=0.8)
            finally:
                ser.close()
        finally:
            httpd.shutdown()
            httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

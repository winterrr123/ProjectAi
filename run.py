import os
import socket
import subprocess
import sys

# Tự động ưu tiên chạy qua môi trường ảo .venv trên ổ D (nơi đã kích hoạt GPU RTX)
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python) and os.path.normcase(sys.executable) != os.path.normcase(venv_python):
    try:
        sys.exit(subprocess.call([venv_python] + sys.argv))
    except KeyboardInterrupt:
        sys.exit(0)

import uvicorn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    if is_port_in_use(port, host):
        print(f"\n[!] CẢNH BÁO: Cổng {port} đang bị chiếm dụng bởi một tiến trình khác (Server đã chạy sẵn).")
        print(f"[i] Đang tự động chuyển sang cổng dự phòng: {port + 1}\n")
        port += 1

    print(f">> Server đang chạy tại: http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


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


def free_port(port: int) -> bool:
    """Tự động giải phóng cổng nếu bị chiếm bởi tiến trình cũ/treo"""
    try:
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        killed = False
        current_pid = os.getpid()
        for line in output.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = int(parts[-1])
                if pid > 0 and pid != current_pid:
                    print(f"[i] Đang tự động giải phóng cổng {port} từ tiến trình cũ (PID {pid})...")
                    subprocess.call(f"taskkill /F /T /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    killed = True
        return killed
    except Exception:
        return False


if __name__ == "__main__":
    port = 8000
    host = "0.0.0.0"

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    if is_port_in_use(port, "127.0.0.1"):
        print(f"\n[!] Cổng {port} đang bị chiếm dụng. Đang tự động dọn dẹp để giữ đúng cổng {port} cho Google OAuth...")
        free_port(port)
        import time
        time.sleep(1)

    if is_port_in_use(port, "127.0.0.1"):
        print(f"[!] Không thể giải phóng cổng {port}, chuyển sang cổng dự phòng: {port + 1}")
        print("[!] Lưu ý: Chuyển cổng có thể gây lỗi 'origin_mismatch' với Google OAuth nếu Google Cloud chưa đăng ký cổng mới.\n")
        port += 1

    print("=" * 60)
    print(">> Product Vision AI Server đang khởi động...")
    print(f">> Khuyến nghị truy cập Google OAuth: http://localhost:{port}")
    print(f">> Hoặc truy cập qua IP cục bộ:       http://127.0.0.1:{port}")
    print("=" * 60)
    uvicorn.run("app.main:app", host=host, port=port, reload=True)



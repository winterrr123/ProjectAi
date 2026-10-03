import socket
import sys
import uvicorn


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


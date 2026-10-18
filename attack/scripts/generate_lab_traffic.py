import socket
import time
from urllib.request import urlopen

TARGET = "10.10.0.117"
HTTP_PORT = 9999
UDP_PORT = 9998

TCP_REQUESTS = 30
UDP_PACKETS = 30
DELAY = 0.2


def generate_tcp_traffic():
    print(f"Sending {TCP_REQUESTS} HTTP requests to {TARGET}:{HTTP_PORT}")

    for i in range(TCP_REQUESTS):
        try:
            with urlopen(
                f"http://{TARGET}:{HTTP_PORT}/",
                timeout=3
            ) as response:
                response.read(1024)

            print(f"HTTP request {i + 1}/{TCP_REQUESTS} sent")

        except Exception as exc:
            print(f"HTTP request {i + 1} failed: {exc}")

        time.sleep(DELAY)


def generate_udp_traffic():
    print(f"Sending {UDP_PACKETS} UDP test packets to {TARGET}:{UDP_PORT}")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for i in range(UDP_PACKETS):
            message = f"IDS lab test packet {i + 1}".encode()
            sock.sendto(message, (TARGET, UDP_PORT))
            print(f"UDP packet {i + 1}/{UDP_PACKETS} sent")
            time.sleep(DELAY)


if __name__ == "__main__":
    print(f"Target receiver: {TARGET}")
    generate_tcp_traffic()
    generate_udp_traffic()
    print("Traffic generation complete.")

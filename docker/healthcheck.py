# File: docker/healthcheck.py
# Standard library only: importing workshops_ai.settings made every probe cost about 0.4 s of CPU,
# where the request itself costs a few milliseconds. Reads the same variables the settings do.
import http.client
import os
import sys


def main() -> int:
    host = os.environ.get("APP_HOST") or os.environ.get("WORKSHOPS_AI_APP_HOST") or "0.0.0.0"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = int(os.environ.get("APP_PORT") or os.environ.get("WORKSHOPS_AI_APP_PORT") or 8000)
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/livez")
        return 0 if conn.getresponse().status == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())

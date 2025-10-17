# File: docker/healthcheck.py
import sys
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from workshops_ai.settings import get_settings


def main() -> int:
    s = get_settings()
    host = "127.0.0.1" if s.app_host in ("0.0.0.0", "::") else s.app_host
    url = f"http://{host}:{s.app_port}/livez"
    try:
        with urlopen(url, timeout=2) as r:
            return 0 if r.status == 200 else 1
    except (URLError, HTTPError, Exception):
        return 1


if __name__ == "__main__":
    sys.exit(main())

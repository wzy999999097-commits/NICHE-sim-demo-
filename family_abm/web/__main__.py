"""
Launch:  python -m family_abm.web
"""
import sys
import webbrowser
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    port = 8520
    url = f'http://127.0.0.1:{port}'

    print(f'\n  Family ABM Dashboard')
    print(f'  {"─" * 42}')
    print(f'  Launching at {url}')
    print(f'  Press Ctrl+C to stop\n')

    threading.Thread(target=lambda: (
        time.sleep(1.5),
        webbrowser.open(url),
    ), daemon=True).start()

    import uvicorn
    uvicorn.run('family_abm.web.app:app', host='127.0.0.1', port=port,
                log_level='warning')


if __name__ == '__main__':
    main()

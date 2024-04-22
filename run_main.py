from __future__ import annotations
import streamlit.web.cli as stcli
import os, sys

def build_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath('.')

    return os.path.join(base_path, relative_path)

def main() -> None:
    sys.argv=[
        'streamlit', 'run', build_path('main.py'),
        '--global.developmentMode=false',
        '--client.showErrorDetails=false',
        '--server.headless=true',
        '--server.runOnSave=false',
        '--server.fileWatcherType=none'
    ]
    sys.exit(stcli.main())

if __name__ == '__main__':
    main()
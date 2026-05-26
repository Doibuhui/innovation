import sys
import os
import time
import webbrowser
import threading
import signal

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

def main():
    from config import Config
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    from server import app

    if os.path.isdir(FRONTEND_DIST):
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        from fastapi import Request

        app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = os.path.join(FRONTEND_DIST, full_path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    print("=" * 50)
    print("  PINN 故障诊断系统")
    print("  访问地址: http://127.0.0.1:8000")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    main()

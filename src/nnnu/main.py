"""nnnu 启动入口：uvicorn 跑 FastAPI app。"""

from __future__ import annotations

import uvicorn

from nnnu.api.main import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

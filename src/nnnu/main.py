"""nnnu 启动入口：uvicorn 跑 FastAPI app。"""

from __future__ import annotations

import logging

import uvicorn

from nnnu.api.main import app


def main() -> None:
    # usage 等业务日志走 INFO，无此配置根 logger 会丢弃 INFO 行
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

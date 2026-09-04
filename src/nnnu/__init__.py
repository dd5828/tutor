"""nnnu — 教育垂直 AI 辅导产品（参考 DeepTutor 架构自研）。

依赖铁律：只能上层调下层。core 禁止 import 任何业务模块；
tools/capabilities 禁止互相 import（通过注册表交互）。
"""

__version__ = "0.1.0"

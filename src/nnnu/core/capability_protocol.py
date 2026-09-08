"""能力协议。

能力层（Level 2）基类：多阶段 agent 管线，用户选定深模式时执行。
CapabilityManifest 只保留 name / description / stages / tools_used 四字段。
"加一个新能力 = 写一个类"，这是产品可扩展性的全部来源。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .bus import StreamBus
from .context import TurnContext


@dataclass
class CapabilityManifest:
    """能力的静态元数据。"""

    name: str
    description: str
    stages: list[str] = field(default_factory=list)  # 前端展示阶段
    tools_used: list[str] = field(default_factory=list)  # 声明用哪些工具


class BaseCapability(ABC):
    """所有能力（深模式）的抽象基类。

    子类必须提供 ``manifest`` 并实现 ``run``。

    Example::

        class MySolverCapability(BaseCapability):
            manifest = CapabilityManifest(
                name="my_solver",
                description="引导学生自己得出答案。",
                stages=["诊断", "引导", "讲解"],
                tools_used=["rag", "check_answer"],
            )

            async def run(self, context, bus):
                async with bus.stage("诊断", source=self.name):
                    ...
    """

    manifest: CapabilityManifest

    @abstractmethod
    async def run(self, context: TurnContext, bus: StreamBus) -> None:
        """执行完整能力管线，事件发往 *bus*。"""
        ...

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def stages(self) -> list[str]:
        return self.manifest.stages

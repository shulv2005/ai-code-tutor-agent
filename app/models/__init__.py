"""SQLAlchemy ORM 模型包。

后续步骤在此注册模型（Issue / AgentRun / TraceRecord 等），统一继承
`app.core.database.Base`。导入本包即完成元数据注册，`init_db()` 会据此自动建表。

新增模型时务必在此导出，否则 create_all 不会建表。
"""

from app.models.code import CodeFile, CodeSymbol
from app.models.code_check import CodeCheckRecord
from app.models.code_fix import CodeFixRecord
from app.models.comment import CommentRecord
from app.models.file_record import ClassifiedFileRecord
from app.models.repository import Repository, RepoStatus, utc_now
from app.models.tutor import TutorAction, TutorRecord

__all__ = [
    "ClassifiedFileRecord",
    "CodeCheckRecord",
    "CodeFile",
    "CodeFixRecord",
    "CodeSymbol",
    "CommentRecord",
    "RepoStatus",
    "Repository",
    "TutorAction",
    "TutorRecord",
    "utc_now",
]

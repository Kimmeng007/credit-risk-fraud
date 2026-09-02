import os
import shutil
from pathlib import Path

_FALLBACK_JAVA_HOMES = [
    Path("C:/Program Files/Eclipse Adoptium"),
]


def _discover_java_home() -> str | None:
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    if shutil.which("java"):
        return None
    for base in _FALLBACK_JAVA_HOMES:
        if base.is_dir():
            candidates = sorted(base.glob("jdk-*"), reverse=True)
            if candidates:
                return str(candidates[0])
    return None


def pytest_configure(config):
    java_home = _discover_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")

    hadoop_home = Path("C:/hadoop")
    if hadoop_home.is_dir() and "HADOOP_HOME" not in os.environ:
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")

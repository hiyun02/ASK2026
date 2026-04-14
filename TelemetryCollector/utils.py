import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

def utc_now_iso() -> str:
    """현재 시간을 UTC ISO 8601 형식으로 반환"""
    return datetime.now(timezone.utc).isoformat()

def safe_int(value: Any) -> Optional[int]:
    """다양한 형태의 입력을 안전하게 정수로 변환"""
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None

def safe_float(value: Any) -> Optional[float]:
    """다양한 형태의 입력을 안전하게 실수로 변환 (실패 시 None)"""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

def setup_logging(log_path: Optional[str]) -> None:
    """콘솔 및 파일 출력을 위한 로깅 환경 설정"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        log_file = Path(log_path)
        # 로그 저장 폴더가 없으면 생성
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
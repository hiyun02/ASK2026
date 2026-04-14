from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class CollectorConfig:
    """수집기 실행 설정을 관리하는 데이터 클래스"""
    redis_host: str
    redis_port: int
    redis_db: int
    redis_password: Optional[str]
    stream_key: str
    group_name: str
    run_id: str
    output_path: str
    output_format: str = "parquet"  # parquet | csv
    sampling_interval: float = 1.0
    flush_interval: int = 10
    log_path: Optional[str] = None
    collect_optional_metrics: bool = False
    replay_start_time: Optional[str] = None

@dataclass
class TelemetryRow:
    """논문용 최소 다변량 데이터셋의 한 행(Row) 규격"""
    run_id: str
    replay_time: str
    sample_index: int
    sampling_interval_sec: float

    # 핵심 지표만 유지
    event_count: Optional[int] = None          # 입력 부하
    group_lag: Optional[int] = None            # 병목 핵심 지표
    pending_count: Optional[int] = None        # 처리 중/미ACK 적체
    used_memory_bytes: Optional[int] = None    # Redis 메모리 사용량
    host_cpu_util_pct: Optional[float] = None  # Redis host CPU 사용률
    host_mem_util_pct: Optional[float] = None  # Redis host 메모리 사용률


@dataclass
class RunMeta:
    """전체 실험의 통계 및 메타데이터를 관리"""
    run_id: str
    started_at: str
    ended_at: Optional[str] = None
    sample_count: int = 0
    partial_failure_count: int = 0
    full_failure_count: int = 0
    null_lag_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
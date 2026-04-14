import json
import logging
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import psutil
import redis

from config import CollectorConfig, TelemetryRow, RunMeta
from utils import utc_now_iso, safe_int, safe_float


class RedisTelemetryCollector:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
        )

        self.should_stop = False
        self.buffer: List[Dict[str, Any]] = []
        self.sample_index = 0

        # event_count 계산용
        self.prev_entries_added: Optional[int] = None

        self.run_meta = RunMeta(
            run_id=config.run_id,
            started_at=utc_now_iso(),
            config=asdict(config),
        )

        self.output_path = Path(config.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_path.with_suffix(".meta.json")

    def stop(self, *_args: Any) -> None:
        logging.info("Stop signal received. Finishing collector...")
        self.should_stop = True

    def validate_environment(self) -> None:
        logging.info("Validating Redis connection...")
        self.redis_client.ping()

        logging.info("Checking stream existence / accessibility...")
        try:
            self.redis_client.xinfo_stream(self.config.stream_key)
        except redis.ResponseError as e:
            raise RuntimeError(
                f"Cannot access stream '{self.config.stream_key}'. "
                f"Ensure stream exists before starting collector. Detail: {e}"
            ) from e

        logging.info("Checking group existence...")
        groups = self.redis_client.xinfo_groups(self.config.stream_key)
        group_names = {g.get("name") for g in groups}
        if self.config.group_name not in group_names:
            raise RuntimeError(
                f"Group '{self.config.group_name}' not found in stream '{self.config.stream_key}'."
            )

        logging.info("Environment validation completed.")

    def fetch_pending_count_fallback(self) -> Optional[int]:
        try:
            result = self.redis_client.xpending(self.config.stream_key, self.config.group_name)
            if isinstance(result, dict):
                return safe_int(result.get("pending"))
            if isinstance(result, (list, tuple)) and len(result) >= 1:
                return safe_int(result[0])
            return None
        except Exception:
            return None

    def collect_one(self) -> TelemetryRow:
        replay_time = utc_now_iso()

        row = TelemetryRow(
            run_id=self.config.run_id,
            replay_time=replay_time,
            sample_index=self.sample_index,
            sampling_interval_sec=self.config.sampling_interval,
        )

        partial_errors: List[str] = []

        # 1) 호스트 자원 수집
        try:
            row.host_cpu_util_pct = psutil.cpu_percent(interval=None)
            row.host_mem_util_pct = safe_float(psutil.virtual_memory().percent)
        except Exception as e:
            partial_errors.append(f"psutil_error={e!s}")

        # event_count를 통한 stream 정보 수집
        try:
            stream_info = self.redis_client.xinfo_stream(self.config.stream_key)
            entries_added = safe_int(stream_info.get("entries-added"))

            if entries_added is not None and self.prev_entries_added is not None:
                diff = entries_added - self.prev_entries_added
                row.event_count = max(diff, 0)
            else:
                row.event_count = 0

            self.prev_entries_added = entries_added
        except Exception as e:
            partial_errors.append(f"xinfo_stream_error={e!s}")

        # lag, pending를 통한 group 정보 수집
        try:
            groups = self.redis_client.xinfo_groups(self.config.stream_key)
            group_info = next((g for g in groups if g.get("name") == self.config.group_name), None)

            if group_info:
                row.group_lag = safe_int(group_info.get("lag"))
                if row.group_lag is None:
                    self.run_meta.null_lag_count += 1

                row.pending_count = safe_int(group_info.get("pending"))
            else:
                partial_errors.append("group_info_missing")
        except Exception as e:
            partial_errors.append(f"xinfo_groups_error={e!s}")

        if row.pending_count is None:
            row.pending_count = self.fetch_pending_count_fallback()

        # memory 정보 수집
        try:
            memory_info = self.redis_client.info(section="memory")
            row.used_memory_bytes = safe_int(memory_info.get("used_memory"))
        except Exception as e:
            partial_errors.append(f"info_memory_error={e!s}")

        if partial_errors:
            self.run_meta.partial_failure_count += 1
            logging.warning("Partial collection failure: %s", " | ".join(partial_errors))

        return row

    def flush_buffer(self) -> None:
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)

        if self.config.output_format.lower() == "csv":
            header = not self.output_path.exists()
            df.to_csv(self.output_path, mode="a", index=False, header=header)
        else:
            if self.output_path.exists():
                old_df = pd.read_parquet(self.output_path)
                df = pd.concat([old_df, df], ignore_index=True)
            df.to_parquet(self.output_path, index=False)

        logging.info("Flushed %d rows to %s", len(self.buffer), self.output_path)
        self.buffer.clear()
        self.write_meta()

    def write_meta(self) -> None:
        self.run_meta.sample_count = self.sample_index
        self.meta_path.write_text(
            json.dumps(asdict(self.run_meta), indent=2),
            encoding="utf-8"
        )

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        self.validate_environment()

        psutil.cpu_percent(interval=None)  # warm-up
        next_tick = time.monotonic()

        while not self.should_stop:
            try:
                row = self.collect_one()
            except Exception as e:
                logging.exception("Full collection failure: %s", e)
                row = TelemetryRow(
                    run_id=self.config.run_id,
                    replay_time=utc_now_iso(),
                    sample_index=self.sample_index,
                    sampling_interval_sec=self.config.sampling_interval,
                )
                self.run_meta.full_failure_count += 1

            self.buffer.append(asdict(row))
            self.sample_index += 1

            if len(self.buffer) >= self.config.flush_interval:
                self.flush_buffer()

            next_tick += self.config.sampling_interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                logging.warning("Collector loop lagging behind by %.3f sec", abs(sleep_for))

        self.flush_buffer()
        self.run_meta.ended_at = utc_now_iso()
        self.write_meta()
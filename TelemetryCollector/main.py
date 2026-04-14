import argparse
import sys
from config import CollectorConfig
from utils import setup_logging
from engine import RedisTelemetryCollector

def parse_args() -> CollectorConfig:
    parser = argparse.ArgumentParser(description="Redis Streams Telemetry Collector")

    # Redis 설정
    parser.add_argument("--redis-host", required=True)
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-password", default=None)

    # 스트림 설정
    parser.add_argument("--stream-key", required=True)
    parser.add_argument("--group-name", required=True)

    # 실험 및 출력 설정
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-format", default="parquet", choices=["parquet", "csv"])

    # 간격 설정
    parser.add_argument("--sampling-interval", type=float, default=1.0)
    parser.add_argument("--flush-interval", type=int, default=10)

    # 기타
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--collect-optional-metrics", action="store_true")

    args = parser.parse_args()

    return CollectorConfig(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_password=args.redis_password,
        stream_key=args.stream_key,
        group_name=args.group_name,
        run_id=args.run_id,
        output_path=args.output_path,
        output_format=args.output_format,
        sampling_interval=args.sampling_interval,
        flush_interval=args.flush_interval,
        log_path=args.log_path,
        collect_optional_metrics=args.collect_optional_metrics
    )

def main() -> None:
    config = parse_args()
    setup_logging(config.log_path)

    collector = RedisTelemetryCollector(config)
    try:
        collector.run()
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
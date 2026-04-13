package ICN.itrc_project.lbsproducer.service;

import ICN.itrc_project.lbsproducer.dto.LocationRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.connection.stream.StreamRecords;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
public class LocationEventProducer {
    @Value("${tdrive-dataset.path}")
    private String parquetPath;
    @Value("${redis.stream.name}")
    private String streamKey;
    private final RedisTemplate<String, Object> redisTemplate;


    public LocationEventProducer(RedisTemplate<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    /**
     * Parquet 파일을 읽어서 Replay 수행하는 핵심 메서드
     */
    @Async
    public void startReplay(String startDate, String endDate, double speedup) {
        if (speedup <= 0) {
            throw new IllegalArgumentException("speedup must be > 0");
        }

        try {
            LocalDate start = LocalDate.parse(startDate); // 예: "2008-02-02"
            LocalDate end = LocalDate.parse(endDate);     // 예: "2008-02-07"

            if (end.isBefore(start)) {
                throw new IllegalArgumentException("endDate must be greater than or equal to startDate");
            }

            Path parquetDir = Paths.get(parquetPath);
            if (!Files.exists(parquetDir) || !Files.isDirectory(parquetDir)) {
                throw new IllegalArgumentException("Invalid parquet directory path: " + parquetPath);
            }

            long totalSentCount = 0L;
            long overallReplayStartNano = System.nanoTime();

            log.info(
                    "Replay started. parquetPath={}, startDate={}, endDate={}, speedup={}x",
                    parquetPath, startDate, endDate, speedup
            );

            LocalDate current = start;
            while (!current.isAfter(end)) {
                String fileName = current + ".parquet";   // 예: 2008-02-02.parquet
                Path filePath = parquetDir.resolve(fileName);

                if (!Files.exists(filePath)) {
                    log.warn("Parquet file not found. Skipping file={}", filePath);
                    current = current.plusDays(1);
                    continue;
                }

                long fileSentCount = 0L;
                long fileReplayStartNano = System.nanoTime();
                LocalDateTime firstSourceTime = null;
                LocalDateTime lastSourceTime = null;

                log.info("Daily replay started. file={}", filePath);

                Table data = Table.read().file(filePath.toString());

                for (Row row : data) {
                    LocationRequest request = new LocationRequest(
                            row.getInt("taxi_id"),
                            row.getDouble("longitude"),
                            row.getDouble("latitude"),
                            row.getDateTime("timestamp")
                    );

                    if (firstSourceTime == null) {
                        firstSourceTime = request.timestamp();
                        lastSourceTime = request.timestamp();
                    }

                    // 첫 이벤트 기준 원본 경과 시간
                    Duration sourceElapsed = Duration.between(firstSourceTime, request.timestamp());
                    long sourceElapsedMillis = sourceElapsed.toMillis();

                    // 배속 적용 후 목표 replay 경과 시간
                    long targetReplayElapsedMillis = (long) (sourceElapsedMillis / speedup);

                    // 실제 replay 시작 후 경과 시간
                    long actualReplayElapsedMillis = (System.nanoTime() - fileReplayStartNano) / 1_000_000L;

                    // 목표 시각까지 필요한 만큼만 sleep
                    long sleepMillis = targetReplayElapsedMillis - actualReplayElapsedMillis;
                    if (sleepMillis > 0) {
                        Thread.sleep(sleepMillis);
                    }

                    sendToRedis(request);

                    fileSentCount++;
                    totalSentCount++;
                    lastSourceTime = request.timestamp();

                    if (fileSentCount % 100000 == 0) {
                        log.info(
                                "Daily replay progress. file={}, fileSentCount={}, totalSentCount={}",
                                filePath, fileSentCount, totalSentCount
                        );
                    }
                }

                long fileReplayDurationMillis = (System.nanoTime() - fileReplayStartNano) / 1_000_000L;

                log.info(
                        "Daily replay completed. file={}, fileSentCount={}, firstSourceTime={}, lastSourceTime={}, replayDurationSec={}",
                        filePath,
                        fileSentCount,
                        firstSourceTime,
                        lastSourceTime,
                        fileReplayDurationMillis / 1000.0
                );

                current = current.plusDays(1);
            }

            long overallReplayDurationMillis = (System.nanoTime() - overallReplayStartNano) / 1_000_000L;

            log.info(
                    "Replay completed. startDate={}, endDate={}, totalSentCount={}, totalReplayDurationSec={}",
                    startDate,
                    endDate,
                    totalSentCount,
                    overallReplayDurationMillis / 1000.0
            );

        } catch (Exception e) {
            log.error(
                    "Replay failed. parquetPath={}, startDate={}, endDate={}, speedup={}x",
                    parquetPath, startDate, endDate, speedup, e
            );
        }
    }
    private void sendToRedis(LocationRequest request) {
        Map<String, String> fields = new HashMap<>();
        fields.put("taxi_id", String.valueOf(request.taxi_id()));
        fields.put("longitude", String.valueOf(request.longitude()));
        fields.put("latitude", String.valueOf(request.latitude()));
        fields.put("source_time", request.timestamp().toString());

        MapRecord<String, String, String> record = StreamRecords.newRecord()
                .in(streamKey)
                .ofMap(fields);
        this.redisTemplate.opsForStream().add(record);
    }
}
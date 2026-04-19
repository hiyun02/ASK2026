package ICN.itrc_project.lbsproducer.service;

import ICN.itrc_project.lbsproducer.dto.LocationRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.avro.generic.GenericRecord;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.avro.AvroParquetReader;
import org.apache.parquet.hadoop.ParquetReader;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.connection.stream.StreamRecords;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.*;
import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class LocationEventProducer {

    @Value("${tdrive-dataset.path}")
    private String parquetPath;
    @Value("${redis.stream.name}")
    private String streamKey;
    private final RedisTemplate<String, Object> redisTemplate;

    @Async
    public void startReplay(String startDate, String endDate, double speedMultiplier) {

        if (speedMultiplier <= 0) {
            throw new IllegalArgumentException("speedMultiplier must be > 0");
        }

        try {
            LocalDate start = LocalDate.parse(startDate);
            LocalDate end = LocalDate.parse(endDate);

            if (end.isBefore(start)) {
                throw new IllegalArgumentException("endDate must be >= startDate");
            }

            long totalSentCount = 0L;
            long overallReplayStartNano = System.nanoTime();

            log.info("Replay started. parquetDirPath={}, startDate={}, endDate={}, speedMultiplier={}x",
                    parquetPath, startDate, endDate, speedMultiplier);

            LocalDate current = start;
            while (!current.isAfter(end)) {
                String fileName = current + ".parquet";
                java.nio.file.Path filePath = Paths.get(parquetPath, fileName);

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
                Configuration conf = new Configuration();
                try (ParquetReader<GenericRecord> reader =
                             AvroParquetReader.<GenericRecord>builder(new Path(filePath.toString()))
                                     .withConf(conf)
                                     .build()) {

                    GenericRecord record;
                    while ((record = reader.read()) != null) {
                        LocationRequest request = new LocationRequest(
                                toInteger(record.get("taxi_id")),
                                toDouble(record.get("longitude")),
                                toDouble(record.get("latitude")),
                                toLocalDateTime(record.get("timestamp"))
                        );

                        if (firstSourceTime == null) {
                            firstSourceTime = request.timestamp();
                            lastSourceTime = request.timestamp();
                        }

                        Duration sourceElapsed = Duration.between(firstSourceTime, request.timestamp());
                        long sourceElapsedMillis = sourceElapsed.toMillis();

                        long targetReplayElapsedMillis = (long) (sourceElapsedMillis / speedMultiplier);
                        long actualReplayElapsedMillis = (System.nanoTime() - fileReplayStartNano) / 1_000_000L;

                        long sleepMillis = targetReplayElapsedMillis - actualReplayElapsedMillis;
                        if (sleepMillis > 0) {
                            Thread.sleep(sleepMillis);
                        }

                        sendToRedis(request);

                        fileSentCount++;
                        totalSentCount++;
                        lastSourceTime = request.timestamp();

                        if (fileSentCount % 100000 == 0) {
                            log.info("Daily replay progress. file={}, fileSentCount={}, totalSentCount={}",
                                    filePath, fileSentCount, totalSentCount);
                        }
                    }
                }

                long fileReplayDurationMillis = (System.nanoTime() - fileReplayStartNano) / 1_000_000L;
                log.info("Daily replay completed. file={}, fileSentCount={}, firstSourceTime={}, lastSourceTime={}, replayDurationSec={}",
                        filePath, fileSentCount, firstSourceTime, lastSourceTime, fileReplayDurationMillis / 1000.0);
                current = current.plusDays(1);

            }

            long overallReplayDurationMillis = (System.nanoTime() - overallReplayStartNano) / 1_000_000L;
            log.info("Replay completed. startDate={}, endDate={}, totalSentCount={}, totalReplayDurationSec={}",
                    startDate, endDate, totalSentCount, overallReplayDurationMillis / 1000.0);

        } catch (Exception e) {
            log.error("Replay failed. parquetDirPath={}, startDate={}, endDate={}, speedMultiplier={}x",
                    parquetPath, startDate, endDate, speedMultiplier, e);
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

        redisTemplate.opsForStream().add(record);
    }

    private Integer toInteger(Object value) {
        if (value == null) return null;
        if (value instanceof Integer i) return i;
        if (value instanceof Long l) return l.intValue();
        return Integer.parseInt(value.toString());
    }

    private Double toDouble(Object value) {
        if (value == null) return null;
        if (value instanceof Double d) return d;
        if (value instanceof Float f) return (double) f;
        if (value instanceof Number n) return n.doubleValue();
        return Double.parseDouble(value.toString());
    }

    private LocalDateTime toLocalDateTime(Object value) {
        if (value == null) {
            throw new IllegalArgumentException("timestamp is null");
        }
        if (value instanceof LocalDateTime ldt) {
            return ldt;
        }
        if (value instanceof Instant instant) {
            return LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
        }

        if (value instanceof Long epochLike) {
            // parquet가 millis인지 micros인지 대략 크기로 판별
            if (epochLike > 1_000_000_000_000_0L) { // micros
                long millis = epochLike / 1000L;
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(millis), ZoneId.systemDefault());
            } else { // millis
                return LocalDateTime.ofInstant(Instant.ofEpochMilli(epochLike), ZoneId.systemDefault());
            }
        }
        return LocalDateTime.parse(value.toString());
    }
}
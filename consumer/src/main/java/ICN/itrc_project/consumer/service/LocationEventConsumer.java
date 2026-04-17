package ICN.itrc_project.consumer.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.stream.StreamListener;
import org.springframework.stereotype.Service;

import java.util.Map;

    @Slf4j
    @Service
    public class LocationEventConsumer implements StreamListener<String, MapRecord<String, String, String>> {

        @Value("${redis.stream.name}")
        private String streamName;

        @Value("${redis.stream.group}")
        private String groupName;

        @Value("${redis.stream.processing-delay-ms:0}")
        private long processingDelayMs;

        private final RedisTemplate<String, Object> redisTemplate;

        public LocationEventConsumer(RedisTemplate<String, Object> redisTemplate) {
            this.redisTemplate = redisTemplate;
        }

        @Override
        public void onMessage(MapRecord<String, String, String> message) {
            try {
                Map<String, String> fields = message.getValue();

                String taxiId = fields.get("taxi_id");
                String longitude = fields.get("longitude");
                String latitude = fields.get("latitude");
                String sourceTime = fields.get("source_time");

                log.debug(
                        "Consumed message. id={}, taxi_id={}, longitude={}, latitude={}, source_time={}",
                        message.getId(), taxiId, longitude, latitude, sourceTime
                );

                // 병목 유도 실험용 선택 지연
                if (processingDelayMs > 0) {
                    Thread.sleep(processingDelayMs);
                }

                // ACK
                redisTemplate.opsForStream().acknowledge(
                        streamName,
                        groupName,
                        message.getId()
                );

            } catch (Exception e) {
                log.error("Failed to process message. id={}", message.getId(), e);
            }
        }
    }
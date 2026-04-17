package ICN.itrc_project.consumer.config;

import ICN.itrc_project.consumer.service.LocationEventConsumer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.connection.stream.Consumer;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.connection.stream.ReadOffset;
import org.springframework.data.redis.connection.stream.StreamOffset;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.StringRedisSerializer;
import org.springframework.data.redis.stream.StreamMessageListenerContainer;
import org.springframework.data.redis.stream.StreamMessageListenerContainer.StreamMessageListenerContainerOptions; // 추가

import java.time.Duration;

@Slf4j
@Configuration
public class RedisConfig {

    @Value("${redis.host}")
    private String host;
    @Value("${redis.port}")
    private int port;
    @Value("${redis.stream.name}")
    private String name;
    @Value("${redis.stream.group}")
    private String group;
    @Value("${redis.stream.consumer-name}")
    private String consumerName;

    @Value("${redis.stream.batch-size:100}")
    private int batchSize;

    @Value("${redis.stream.poll-timeout-ms:1000}")
    private long pollTimeoutMs;

    @Bean
    public RedisConnectionFactory redisConnectionFactory() {
        return new LettuceConnectionFactory(host, port);
    }

    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setHashValueSerializer(new StringRedisSerializer());
        return template;
    }

    @Bean
    public StreamMessageListenerContainer<String, MapRecord<String, String, String>> container(
            RedisConnectionFactory factory, LocationEventConsumer consumer) {

        // 직렬화 방식을 String으로 통일
        StreamMessageListenerContainerOptions<String, MapRecord<String, String, String>> options =
                StreamMessageListenerContainerOptions.builder()
                        .pollTimeout(Duration.ofMillis(pollTimeoutMs))
                        .batchSize(batchSize)
                        .serializer(new StringRedisSerializer())
                        .build();

        StreamMessageListenerContainer<String, MapRecord<String, String, String>> container =
                StreamMessageListenerContainer.create(factory, options);

        container.receive(
                Consumer.from(group, consumerName),
                StreamOffset.create(name, ReadOffset.lastConsumed()),
                consumer
        );

        container.start();
        log.info(">>>>>> Redis Stream Consumer Container Started!");
        return container;
    }
}
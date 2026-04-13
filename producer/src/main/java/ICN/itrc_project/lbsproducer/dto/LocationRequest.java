package ICN.itrc_project.lbsproducer.dto;

import java.time.LocalDateTime;

public record LocationRequest(
        Integer taxi_id,
        Double longitude,
        Double latitude,
        LocalDateTime timestamp
) {}
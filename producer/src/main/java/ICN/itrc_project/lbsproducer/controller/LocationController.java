package ICN.itrc_project.lbsproducer.controller;

import ICN.itrc_project.lbsproducer.service.LocationEventProducer;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class LocationController {

    private final LocationEventProducer eventProducer;

    @GetMapping("/start")
    public String start(@RequestParam double speed,
                        @RequestParam String startDate,
                        @RequestParam String endDate) {
        eventProducer.startReplay(startDate, endDate, speed);
        return "실험 시작! (배속: " + speed + "x)";
    }
}
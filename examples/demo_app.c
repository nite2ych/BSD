// examples/demo_app.c
// BSD alarm system integration demo
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../deployment/board/alarm_engine/alarm_engine.h"

static const char* class_names[] = {"person", "bicycle", "motorcycle", "vehicle"};
static const char* zone_names[]  = {"Zone1", "Zone2", "Zone3"};

// Alarm callback
static void on_alarm(const AlarmEvent* e)
{
    printf("[ALARM] zone=%s class=%s overlap=%.2f frames=%d pos=(%.0f,%.0f)-(%.0f,%.0f)\n",
           zone_names[e->zone_level],
           class_names[e->class_id],
           e->overlap,
           e->frame_count,
           e->x1, e->y1, e->x2, e->y2);
}

int main(int argc, char* argv[])
{
    printf("BSD Alarm Demo\n");

    // 1. Initialize alarm engine
    if (alarm_init("/bsd_detect_shm") != 0) {
        fprintf(stderr, "alarm_init failed\n");
        return 1;
    }

    // 2. Configure 3-level alarm zones (normalized coords 0~1)
    // Zone1 (innermost, red, highest danger)
    alarm_set_zone(0, 0.3f, 0.3f, 0.7f, 0.7f, 0.3f);
    // Zone2 (middle, orange)
    alarm_set_zone(1, 0.15f, 0.15f, 0.85f, 0.85f, 0.3f);
    // Zone3 (outermost, yellow)
    alarm_set_zone(2, 0.0f, 0.0f, 1.0f, 1.0f, 0.3f);

    // 3. Set consecutive frame threshold (10 frames @10fps = 1s)
    alarm_set_frame_threshold(10);

    // 4. Register alarm callback
    alarm_register_callback(on_alarm);

    // 5. Start alarm detection
    alarm_start();
    printf("Alarm engine started. Press Ctrl+C to stop.\n");

    // 6. Main loop (detect engine runs in separate process)
    while (1) {
        sleep(1);
    }

    // 7. Cleanup
    alarm_deinit();
    return 0;
}

// deployment/board/detect_engine/main.c
// BSD main executable: ties detect_engine + alarm_engine together
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include "detect_engine.h"
#include "../alarm_engine/alarm_engine.h"

static volatile int g_running = 1;

static void sig_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void on_alarm(const AlarmEvent* evt)
{
    const char* names[] = {"person", "bicycle", "motorcycle", "vehicle"};
    printf("ALARM: zone=%d class=%s overlap=%.2f frames=%d pos=[%.0f,%.0f,%.0f,%.0f]\n",
           evt->zone_level,
           (evt->class_id >= 0 && evt->class_id < 4) ? names[evt->class_id] : "?",
           evt->overlap, evt->frame_count,
           evt->x1, evt->y1, evt->x2, evt->y2);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.nb> [cam_width cam_height] [conf_thr nms_thr]\n", argv[0]);
        fprintf(stderr, "  Default: 1920x1080 conf=0.3 nms=0.45\n");
        return 1;
    }

    const char* nb_path = argv[1];
    int cam_w = (argc >= 4) ? atoi(argv[2]) : 1920;
    int cam_h = (argc >= 4) ? atoi(argv[3]) : 1080;
    float conf_thr = (argc >= 6) ? atof(argv[4]) : 0.3f;
    float nms_thr  = (argc >= 6) ? atof(argv[5]) : 0.45f;

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    printf("BSD Detect System\n");
    printf("  Model: %s\n", nb_path);
    printf("  Camera: %dx%d\n", cam_w, cam_h);
    printf("  Thresholds: conf=%.2f nms=%.2f\n", conf_thr, nms_thr);

    // Init detect engine
    if (detect_init(nb_path, "/bsd_detect_shm", cam_w, cam_h, conf_thr, nms_thr) != 0) {
        fprintf(stderr, "detect_init failed\n");
        return 1;
    }
    printf("Detect engine initialized\n");

    // Init alarm engine
    if (alarm_init("/bsd_detect_shm") != 0) {
        fprintf(stderr, "alarm_init failed\n");
        detect_deinit();
        return 1;
    }

    // Set alarm zones (normalized coords 0~1)
    // Zone 0: center region (highest priority)
    alarm_set_zone(0, 0.25f, 0.2f, 0.75f, 0.8f, 0.3f);
    // Zone 1: full frame (lower priority)
    alarm_set_zone(1, 0.0f, 0.0f, 1.0f, 1.0f, 0.3f);

    alarm_set_frame_threshold(10);
    alarm_register_callback(on_alarm);

    if (alarm_start() != 0) {
        fprintf(stderr, "alarm_start failed\n");
        alarm_deinit();
        detect_deinit();
        return 1;
    }
    printf("Alarm engine started\n");

    // Allocate frame buffer
    size_t frame_sz = (size_t)cam_w * cam_h * 3;
    uint8_t* frame_buf = (uint8_t*)malloc(frame_sz);
    if (!frame_buf) {
        fprintf(stderr, "malloc frame_buf failed\n");
        alarm_deinit();
        detect_deinit();
        return 1;
    }

    printf("\nReading frames from stdin (raw BGR interleaved %dx%d)...\n", cam_w, cam_h);
    printf("Press Ctrl+C to stop\n\n");

    // Main loop: read frames from stdin, process
    while (g_running) {
        size_t n = fread(frame_buf, 1, frame_sz, stdin);
        if (n != frame_sz) {
            if (n == 0) break;  // EOF
            fprintf(stderr, "Short read: %zu / %zu bytes\n", n, frame_sz);
            break;
        }

        if (detect_process_frame(frame_buf) != 0) {
            fprintf(stderr, "detect_process_frame failed\n");
            continue;
        }

        // Print detection results
        const DetResult* res = detect_get_result();
        if (res && res->count > 0) {
            const char* names[] = {"person", "bicycle", "motorcycle", "vehicle"};
            printf("[frame %u] %u detections:\n", res->frame_id, res->count);
            for (uint32_t i = 0; i < res->count; i++) {
                const DetObject* d = &res->objects[i];
                printf("  %s conf=%.3f bbox=[%.0f,%.0f,%.0f,%.0f]\n",
                       (d->class_id >= 0 && d->class_id < 4) ? names[d->class_id] : "?",
                       d->conf, d->x1, d->y1, d->x2, d->y2);
            }
        } else {
            printf("[frame %u] no detections\n", res ? res->frame_id : 0);
        }
    }

    printf("\nShutting down...\n");
    free(frame_buf);
    alarm_deinit();
    detect_deinit();
    printf("Done.\n");
    return 0;
}

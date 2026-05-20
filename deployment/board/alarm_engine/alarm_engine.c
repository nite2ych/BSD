// deployment/board/alarm_engine/alarm_engine.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include "alarm_engine.h"

// External functions (zone_mgr.c)
extern void zone_mgr_init(void);
extern int  zone_mgr_set(int level, float x1, float y1, float x2, float y2, float overlap_thr);
extern int  zone_mgr_hit_test(const Box* bbox, float* out_overlap);

// External functions (tracker.c)
extern void tracker_init(void);
extern int  tracker_update(const DetObject* dets, int n_det,
                           const int* zone_hits, const float* overlaps,
                           AlarmEvent* out_alarms, int max_alarms,
                           int frame_thr);

// Internal state
static struct {
    int       running;
    int       frame_threshold;  // consecutive frame alarm threshold (default 10)
    pthread_t thread;

    // Shared memory
    int         shm_fd;
    DetResult*  shm_result;

    // Callback
    alarm_callback_t callback;

    // Previous frame id (for frame rate control)
    uint32_t last_frame_id;
} g_alarm;

int alarm_init(const char* shm_name)
{
    memset(&g_alarm, 0, sizeof(g_alarm));
    g_alarm.frame_threshold = 10;

    // Initialize sub-modules
    zone_mgr_init();
    tracker_init();

    // Open detect engine shared memory (read-only)
    g_alarm.shm_fd = shm_open(shm_name, O_RDONLY, 0666);
    if (g_alarm.shm_fd < 0) return -1;

    g_alarm.shm_result = (DetResult*)mmap(NULL, sizeof(DetResult),
                                          PROT_READ, MAP_SHARED,
                                          g_alarm.shm_fd, 0);
    if (g_alarm.shm_result == MAP_FAILED) {
        close(g_alarm.shm_fd);
        return -1;
    }

    return 0;
}

int alarm_set_zone(int level, float x1, float y1, float x2, float y2,
                   float overlap_thr)
{
    return zone_mgr_set(level, x1, y1, x2, y2, overlap_thr);
}

int alarm_set_frame_threshold(int n)
{
    if (n < 1) return -1;
    g_alarm.frame_threshold = n;
    return 0;
}

int alarm_register_callback(alarm_callback_t cb)
{
    if (!cb) return -1;
    g_alarm.callback = cb;
    return 0;
}

// Alarm engine main loop (runs in separate thread)
static void* alarm_loop(void* arg)
{
    (void)arg;

    while (g_alarm.running) {
        // Wait for new frame
        if (g_alarm.shm_result->frame_id == g_alarm.last_frame_id) {
            usleep(5000);  // 5ms
            continue;
        }
        g_alarm.last_frame_id = g_alarm.shm_result->frame_id;

        uint32_t n_det = g_alarm.shm_result->count;
        if (n_det > MAX_DET) n_det = MAX_DET;

        // Zone hit test for each detection
        int   zone_hits[MAX_DET];
        float overlaps[MAX_DET];
        memset(zone_hits, -1, sizeof(zone_hits));
        memset(overlaps, 0, sizeof(overlaps));

        for (uint32_t i = 0; i < n_det; i++) {
            // Skip vehicle (backup class)
            if (g_alarm.shm_result->objects[i].class_id == CLASS_VEHICLE) {
                zone_hits[i] = -1;
                continue;
            }

            Box bbox = {
                g_alarm.shm_result->objects[i].x1,
                g_alarm.shm_result->objects[i].y1,
                g_alarm.shm_result->objects[i].x2,
                g_alarm.shm_result->objects[i].y2,
            };

            float overlap = 0.0f;
            int hit = zone_mgr_hit_test(&bbox, &overlap);
            zone_hits[i] = hit;
            overlaps[i] = overlap;
        }

        // Update tracking and get alarms
        AlarmEvent alarms[16];
        int n_alarms = tracker_update(
            g_alarm.shm_result->objects, n_det,
            zone_hits, overlaps,
            alarms, 16, g_alarm.frame_threshold);

        // Trigger callback
        if (g_alarm.callback && n_alarms > 0) {
            for (int i = 0; i < n_alarms; i++) {
                g_alarm.callback(&alarms[i]);
            }
        }
    }

    return NULL;
}

int alarm_start(void)
{
    if (g_alarm.running) return -1;
    g_alarm.running = 1;
    if (pthread_create(&g_alarm.thread, NULL, alarm_loop, NULL) != 0) {
        g_alarm.running = 0;
        return -1;
    }
    return 0;
}

int alarm_stop(void)
{
    g_alarm.running = 0;
    if (g_alarm.thread) {
        pthread_join(g_alarm.thread, NULL);
    }
    return 0;
}

void alarm_deinit(void)
{
    alarm_stop();
    if (g_alarm.shm_result && g_alarm.shm_result != MAP_FAILED) {
        munmap(g_alarm.shm_result, sizeof(DetResult));
    }
    if (g_alarm.shm_fd > 0) {
        close(g_alarm.shm_fd);
    }
    memset(&g_alarm, 0, sizeof(g_alarm));
}

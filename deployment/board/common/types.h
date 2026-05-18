// deployment/board/common/types.h
#ifndef BSD_TYPES_H
#define BSD_TYPES_H

#include <stdint.h>

#define MAX_DET      64
#define MAX_TRACKS   64
#define MAX_CLASSES  4
#define MAX_ZONES    3

// Category definitions
enum {
    CLASS_PERSON     = 0,
    CLASS_BICYCLE    = 1,
    CLASS_MOTORCYCLE = 2,
    CLASS_VEHICLE    = 3,   // backup, not used in alarm
};

// Alarm levels (0 = highest priority / innermost)
enum {
    ZONE_LEVEL_1 = 0,
    ZONE_LEVEL_2 = 1,
    ZONE_LEVEL_3 = 2,
};

// Rectangle (normalized coordinates 0~1)
typedef struct {
    float x1, y1, x2, y2;
} Box;

// Single detected object
typedef struct {
    int32_t class_id;
    float   conf;
    float   x1, y1, x2, y2;  // bbox (pixel coords mapped back to original image)
} DetObject;

// Detection result (shared memory, updated every frame)
typedef struct {
    uint32_t  frame_id;
    uint32_t  timestamp_ms;
    uint32_t  count;               // valid object count
    DetObject objects[MAX_DET];
} DetResult;

// Alarm event
typedef struct {
    int32_t zone_level;
    int32_t class_id;
    float   overlap;               // actual overlap ratio
    int32_t frame_count;           // consecutive frame hit count
    float   x1, y1, x2, y2;        // target position when alarm triggered
} AlarmEvent;

// Alarm callback type
typedef void (*alarm_callback_t)(const AlarmEvent* event);

#endif // BSD_TYPES_H

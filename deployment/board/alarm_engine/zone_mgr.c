// deployment/board/alarm_engine/zone_mgr.c
#include <string.h>
#include "../common/types.h"

// Zone storage
typedef struct {
    int   active;       // enabled flag
    Box   box;          // normalized coords (0~1)
    float overlap_thr;  // overlap threshold
} ZoneCfg;

static ZoneCfg g_zones[MAX_ZONES];

// Initialize
void zone_mgr_init(void)
{
    memset(g_zones, 0, sizeof(g_zones));
}

// Set zone
int zone_mgr_set(int level, float x1, float y1, float x2, float y2,
                 float overlap_thr)
{
    if (level < 0 || level >= MAX_ZONES) return -1;
    g_zones[level].active = 1;
    g_zones[level].box.x1 = x1;
    g_zones[level].box.y1 = y1;
    g_zones[level].box.x2 = x2;
    g_zones[level].box.y2 = y2;
    g_zones[level].overlap_thr = overlap_thr;
    return 0;
}

// Calculate bbox-to-zone overlap ratio
// Returns: intersection_area / bbox_area, 0 if no intersection
float zone_mgr_overlap(const Box* bbox, int level)
{
    if (!g_zones[level].active) return 0.0f;

    const Box* z = &g_zones[level].box;

    // AABB intersection
    float ix1 = (bbox->x1 > z->x1) ? bbox->x1 : z->x1;
    float iy1 = (bbox->y1 > z->y1) ? bbox->y1 : z->y1;
    float ix2 = (bbox->x2 < z->x2) ? bbox->x2 : z->x2;
    float iy2 = (bbox->y2 < z->y2) ? bbox->y2 : z->y2;

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;

    float inter_area = iw * ih;
    float bbox_area = (bbox->x2 - bbox->x1) * (bbox->y2 - bbox->y1);
    if (bbox_area <= 0.0f) return 0.0f;

    return inter_area / bbox_area;
}

// Hit test: return highest priority zone level hit, -1 if none
// Level 0 (ZONE_LEVEL_1) has highest priority
int zone_mgr_hit_test(const Box* bbox, float* out_overlap)
{
    for (int level = 0; level < MAX_ZONES; level++) {
        if (!g_zones[level].active) continue;
        float overlap = zone_mgr_overlap(bbox, level);
        if (overlap >= g_zones[level].overlap_thr) {
            if (out_overlap) *out_overlap = overlap;
            return level;
        }
    }
    return -1;
}

// Get zone config
const ZoneCfg* zone_mgr_get(int level)
{
    if (level < 0 || level >= MAX_ZONES || !g_zones[level].active)
        return NULL;
    return &g_zones[level];
}

#ifndef TRILAT_H
#define TRILAT_H

#ifdef __cplusplus
extern "C" {
#endif

#define TRILAT_ANCHOR_COUNT 3

typedef enum {
    TRILAT_OK = 0,
    TRILAT_ERR_NULL = -1,
    TRILAT_ERR_INVALID_INPUT = -2,
    TRILAT_ERR_POOR_GEOMETRY = -3,
    TRILAT_ERR_NO_CONVERGENCE = -4,
} trilat_status_t;

typedef struct {
    double lat_deg;
    double lon_deg;
    double alt_m;
    double distance_m;
    const char *node_id;
} trilat_anchor_t;

typedef struct {
    double lat_deg;
    double lon_deg;
    double alt_m;
    double rms_error_m;
    double max_abs_error_m;
    double residuals_m[TRILAT_ANCHOR_COUNT];
    double expected_distances_m[TRILAT_ANCHOR_COUNT];
    int iterations;
    double geometry_condition;
} trilat_result_t;

trilat_status_t trilat_solve_3_anchor_altitude(
    const trilat_anchor_t anchors[TRILAT_ANCHOR_COUNT],
    double target_alt_m,
    trilat_result_t *result
);

const char *trilat_status_name(trilat_status_t status);

#ifdef __cplusplus
}
#endif

#endif

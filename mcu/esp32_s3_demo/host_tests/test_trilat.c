#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "trilat.h"

static const trilat_anchor_t SAMPLE_ANCHORS[TRILAT_ANCHOR_COUNT] = {
    {
        .lat_deg = 50.4501,
        .lon_deg = 30.5234,
        .alt_m = 180.0,
        .distance_m = 394.1347351919918,
        .node_id = "A",
    },
    {
        .lat_deg = 50.4565,
        .lon_deg = 30.5201,
        .alt_m = 190.0,
        .distance_m = 621.9571290990954,
        .node_id = "B",
    },
    {
        .lat_deg = 50.4510,
        .lon_deg = 30.5340,
        .alt_m = 175.0,
        .distance_m = 553.3913539405108,
        .node_id = "C",
    },
};

static void test_exact_fixture(void)
{
    trilat_result_t result;
    const trilat_status_t status = trilat_solve_3_anchor_altitude(SAMPLE_ANCHORS, 183.5, &result);

    printf("status=%s\n", trilat_status_name(status));
    printf("lat=%.12f lon=%.12f alt=%.6f rms=%.12f max_abs=%.12f iterations=%d condition=%.6f\n",
           result.lat_deg,
           result.lon_deg,
           result.alt_m,
           result.rms_error_m,
           result.max_abs_error_m,
           result.iterations,
           result.geometry_condition);

    assert(status == TRILAT_OK);
    assert(fabs(result.lat_deg - 50.4529) < 1.0e-8);
    assert(fabs(result.lon_deg - 30.5268) < 1.0e-8);
    assert(fabs(result.alt_m - 183.5) < 1.0e-12);
    assert(result.rms_error_m < 1.0e-5);
    assert(result.geometry_condition > 1.0);
}

static void test_invalid_input(void)
{
    trilat_result_t result;
    trilat_anchor_t anchors[TRILAT_ANCHOR_COUNT] = {
        SAMPLE_ANCHORS[0],
        SAMPLE_ANCHORS[1],
        SAMPLE_ANCHORS[2],
    };
    anchors[0].distance_m = -1.0;

    assert(trilat_solve_3_anchor_altitude(anchors, 183.5, &result) == TRILAT_ERR_INVALID_INPUT);
    assert(trilat_solve_3_anchor_altitude(NULL, 183.5, &result) == TRILAT_ERR_NULL);
    assert(trilat_solve_3_anchor_altitude(SAMPLE_ANCHORS, 183.5, NULL) == TRILAT_ERR_NULL);
}

int main(void)
{
    test_exact_fixture();
    test_invalid_input();
    puts("C trilateration tests passed");
    return 0;
}

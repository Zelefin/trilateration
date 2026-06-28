#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "trilat.h"

static const char *TAG = "trilat_demo";

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

static const double EXPECTED_LAT_DEG = 50.4529;
static const double EXPECTED_LON_DEG = 30.5268;
static const double TARGET_ALT_M = 183.5;

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-S3 trilateration self-test");

    trilat_result_t result;
    const trilat_status_t status = trilat_solve_3_anchor_altitude(SAMPLE_ANCHORS, TARGET_ALT_M, &result);
    ESP_LOGI(TAG, "status=%s", trilat_status_name(status));

    if (status == TRILAT_OK) {
        ESP_LOGI(TAG, "expected lat=%.8f lon=%.8f alt=%.2f m", EXPECTED_LAT_DEG, EXPECTED_LON_DEG, TARGET_ALT_M);
        ESP_LOGI(TAG, "actual   lat=%.8f lon=%.8f alt=%.2f m", result.lat_deg, result.lon_deg, result.alt_m);
        ESP_LOGI(TAG, "rms=%.9f m max_abs=%.9f m iterations=%d geometry_condition=%.3f",
                 result.rms_error_m,
                 result.max_abs_error_m,
                 result.iterations,
                 result.geometry_condition);
        for (int i = 0; i < TRILAT_ANCHOR_COUNT; ++i) {
            ESP_LOGI(TAG,
                     "anchor %s observed=%.9f m expected=%.9f m residual=%+.9f m",
                     SAMPLE_ANCHORS[i].node_id,
                     SAMPLE_ANCHORS[i].distance_m,
                     result.expected_distances_m[i],
                     result.residuals_m[i]);
        }
    }

    const bool pass = status == TRILAT_OK &&
                      fabs(result.lat_deg - EXPECTED_LAT_DEG) < 1.0e-8 &&
                      fabs(result.lon_deg - EXPECTED_LON_DEG) < 1.0e-8 &&
                      fabs(result.alt_m - TARGET_ALT_M) < 1.0e-9 &&
                      result.rms_error_m < 1.0e-5;

    ESP_LOGI(TAG, "SELF_TEST_%s", pass ? "PASS" : "FAIL");

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

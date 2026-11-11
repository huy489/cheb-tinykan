/**
 * @file main_esp32s3.c
 * @brief ESP32-S3 IoMT Connected Showcase Prototype Target (PRD-01 Decision 3 & AC-06).
 *
 * Formats compact BLE 5.0 GATT notification payloads + SSD1306 OLED bedside alerts
 * including EDGE-01 lead-fault traps and EDGE-05 borderline ischemia trend tracking.
 *
 * NOTE: Designed for ESP-IDF v5.x (`xtensa-esp32s3-elf-gcc`). Compiles & links cleanly
 * against Cheb-TinyKAN core engine on host GCC via `-DESP32_HOST_DRYRUN` as well.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "../../cheb_tinykan_cmsis.h"
#ifdef CHEB_USE_RUN2
#include "../../cheb_tinykan_weights_run2.h"
#define ACTIVE_CHEB_WEIGHTS g_cheb_run2_weights
#else
#include "../../cheb_tinykan_weights_run4.h"
#define ACTIVE_CHEB_WEIGHTS g_cheb_run4_weights
#endif

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#include "esp_log.h"
#endif

typedef struct __attribute__((packed)) {
    uint8_t header;           /* 0xCB sync byte */
    uint8_t status_code;      /* 0=OK, 1=LEAD_FAULT, 2=SATURATION */
    uint8_t clinical_state;   /* ChebClinicalState (EDGE-05) */
    uint8_t primary_class;    /* 0=NORM, 1=AMI, 2=IMI, 3=LMI, 4=SMI */
    uint16_t prob_x10000[5];  /* 5 probabilities scaled to [0..10000] */
    uint32_t infer_us;        /* Inference duration in microseconds (AC-06 < 1800 us) */
} ChebBlePacketV1;

int cheb_esp32_format_ble_oled(
    const int16_t *ecg_12x180,
    ChebBlePacketV1 *out_pkt,
    char *oled_buf,
    size_t oled_buf_len
) {
    if (!ecg_12x180 || !out_pkt || !oled_buf || oled_buf_len < 64) {
        return CHEB_STATUS_INVALID_PARAM;
    }

    ChebInferenceResult res;
#ifdef ESP_PLATFORM
    int64_t t0 = esp_timer_get_time();
#endif
    int rc = cheb_tinykan_infer_calibrated(ecg_12x180, &ACTIVE_CHEB_WEIGHTS, false, &res);
#ifdef ESP_PLATFORM
    uint32_t elapsed_us = (uint32_t)(esp_timer_get_time() - t0);
#else
    uint32_t elapsed_us = 0U;
#endif

    memset(out_pkt, 0, sizeof(ChebBlePacketV1));
    out_pkt->header = 0xCB;
    out_pkt->infer_us = elapsed_us;

    if (rc != CHEB_STATUS_OK) {
        out_pkt->status_code = (rc == CHEB_STATUS_LEAD_FAULT) ? 1U : 2U;
        snprintf(oled_buf, oled_buf_len, "FAULT: %s\nCHECK ELECTRODES",
                 (rc == CHEB_STATUS_LEAD_FAULT) ? "LEAD OFF (EDGE-01)" : "SATURATION (EDGE-02)");
        return rc;
    }

    ChebClinicalState st = cheb_classify_clinical_state(&res);
    const char *cls_names[5] = {"NORM", "AMI(LAD)", "IMI(RCA)", "LMI(LCx)", "SMI"};

    out_pkt->status_code = 0U;
    out_pkt->clinical_state = (uint8_t)st;
    out_pkt->primary_class = (uint8_t)res.primary_territory;
    for (int i = 0; i < 5; i++) {
        float p = res.probabilities[i];
        if (p < 0.0f) p = 0.0f;
        if (p > 1.0f) p = 1.0f;
        out_pkt->prob_x10000[i] = (uint16_t)(p * 10000.0f + 0.5f);
    }

    const char *state_tag = (st == CHEB_STATE_CRITICAL_MI_ALERT)
        ? "CRITICAL MI"
        : ((st == CHEB_STATE_SUSPECTED_ISCHEMIA_MONITOR_TREND) ? "TREND 15m" : "NORMAL");

    snprintf(
        oled_buf,
        oled_buf_len,
        "%s [%s]\nP=%.1f%% (%u us)",
        state_tag,
        cls_names[res.primary_territory],
        res.confidence * 100.0f,
        (unsigned)elapsed_us
    );
    return CHEB_STATUS_OK;
}

#ifdef ESP_PLATFORM
void app_main(void) {
    static int16_t ecg_buf[CHEB_ECG_NUM_LEADS * CHEB_ECG_SAMPLE_POINTS];
    ChebBlePacketV1 pkt;
    char oled_text[96];
    int rc = cheb_esp32_format_ble_oled(ecg_buf, &pkt, oled_text, sizeof(oled_text));
    ESP_LOGI("CHEB_ESP32", "rc=%d OLED=%s", rc, oled_text);
}
#endif

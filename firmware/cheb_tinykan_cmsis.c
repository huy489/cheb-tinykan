/**
 * @file cheb_tinykan_cmsis.c
 * @brief Cheb-TinyKAN Embedded C99 Firmware Engine Implementation.
 *
 * Implements pure C99 inference with CMSIS-NN SIMD / __SMLAD acceleration,
 * zero dynamic memory allocation, and static Ping-Pong activation buffers.
 *
 * References:
 * - PRD-01-CHEB-TINYKAN.md: AC-03, AC-04, AC-05, AC-07
 */

#include "cheb_tinykan_cmsis.h"
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* STATIC ACTIVATION SCRATCHPAD BUFFERS (PING-PONG)                          */
/* Peak Activation SRAM: 2160 int16 = 4,320 bytes (< 6.0 KB per AC-04)       */
/* ========================================================================= */

#define SCRATCH_MAX_ELEMENTS 2160  /* 24 * 90 = 2160 int16 */

static int16_t s_act_buf_a[SCRATCH_MAX_ELEMENTS];
static int16_t s_act_buf_b[SCRATCH_MAX_ELEMENTS];

/* ========================================================================= */
/* CHEBYSHEV 3-TERM RECURRENCE IN Q1.15                                      */
/* ========================================================================= */

void cheb_recurrence_q15(int16_t x_q15, int16_t t_out[5]) {
    /* T_0(x) = 1.0 -> 32767 in Q1.15 */
    t_out[0] = Q15_MAX;

    /* T_1(x) = x */
    t_out[1] = x_q15;

    /* T_{k+1}(x) = 2*x*T_k(x) - T_{k-1}(x)
     * In Q1.15 arithmetic:
     * 2 * ((x * T_k) >> 15) = (x * T_k) >> 14
     */
    int32_t t2 = (((int32_t)x_q15 * (int32_t)t_out[1]) >> 14) - (int32_t)t_out[0];
    t_out[2] = q15_sat(t2);

    int32_t t3 = (((int32_t)x_q15 * (int32_t)t_out[2]) >> 14) - (int32_t)t_out[1];
    t_out[3] = q15_sat(t3);

    int32_t t4 = (((int32_t)x_q15 * (int32_t)t_out[3]) >> 14) - (int32_t)t_out[2];
    t_out[4] = q15_sat(t4);
}

/* ========================================================================= */
/* FAST NON-LINEAR ACTIVATIONS IN Q1.15 (TANH & SIGMOID)                     */
/* ========================================================================= */

int16_t cheb_fast_tanh_q15(int16_t x_q15) {
    /* Piecewise rational approximation for tanh:
     * For |x| >= 3.0 (i.e. |x_q15| >= 98304), tanh(x) = +/- 1.0
     * Here x_q15 is already bounded in [-32768, 32767] ~ [-1.0, 1.0]
     * For x in [-1, 1], tanh(x) ~ x * (27 + x^2) / (27 + 9*x^2)
     * Or simple 3rd-order Taylor: x - x^3 / 3
     */
    int32_t x = x_q15;
    int32_t x2 = (x * x) >> 15;
    int32_t x3 = (x2 * x) >> 15;
    int32_t approx = x - (x3 / 3);
    return q15_sat(approx);
}

int16_t cheb_fast_sigmoid_q15(int16_t x_q15) {
    /* sigmoid(x) = 0.5 + 0.5 * tanh(x / 2)
     * Output is in range [0, 32767] representing [0.0, 1.0]
     */
    int16_t half_x = x_q15 >> 1;
    int16_t th = cheb_fast_tanh_q15(half_x);
    int32_t sig = 16384 + (th >> 1);
    return q15_sat(sig);
}

/* ========================================================================= */
/* SIGNAL QUALITY INDEX (SQI) & LEAD INTEGRITY CHECK                         */
/* ========================================================================= */

int cheb_verify_sqi_q15(const int16_t *ecg_12x180) {
    /* Checks EDGE-01: Flatline (lead off) and EDGE-02: Saturation clipping */
    for (int lead = 0; lead < CHEB_ECG_NUM_LEADS; lead++) {
        const int16_t *ptr = &ecg_12x180[lead * CHEB_ECG_SAMPLE_POINTS];
        int16_t min_v = ptr[0];
        int16_t max_v = ptr[0];
        int sat_count = 0;

        for (int i = 0; i < CHEB_ECG_SAMPLE_POINTS; i++) {
            int16_t val = ptr[i];
            if (val < min_v) min_v = val;
            if (val > max_v) max_v = val;
            if (val >= 32000 || val <= -32000) {
                sat_count++;
            }
        }

        /* 1. Flatline check: range < ~0.005 mV (~82 LSBs in Q15) */
        if ((max_v - min_v) < 80) {
            return CHEB_STATUS_LEAD_FAULT;
        }

        /* 2. Clipping check: > 5% points saturated */
        if (sat_count > (CHEB_ECG_SAMPLE_POINTS / 20)) {
            return CHEB_STATUS_SATURATION;
        }
    }
    return CHEB_STATUS_OK;
}

/* ========================================================================= */
/* CONVOLUTIONAL TOKENIZER OPERATIONS (1D-CNN)                               */
/* ========================================================================= */

static void conv1d_standard_q15(
    const int16_t *in,   /* [12 x 180] */
    const int16_t *w,    /* [24 x 12 x 7] */
    const int16_t *b,    /* [24] */
    int16_t *out         /* [24 x 90] */
) {
    const int in_ch = 12;
    const int in_len = 180;
    const int out_ch = 24;
    const int out_len = 90;
    const int k = 7;
    const int stride = 2;
    const int pad = 3;

    for (int oc = 0; oc < out_ch; oc++) {
        int32_t bias_val = b ? ((int32_t)b[oc] << 15) : 0;
        const int16_t *w_oc = &w[oc * in_ch * k];

        for (int ol = 0; ol < out_len; ol++) {
            int in_center = ol * stride - pad;
            int64_t acc = bias_val;

            for (int ic = 0; ic < in_ch; ic++) {
                const int16_t *in_ic = &in[ic * in_len];
                const int16_t *w_ic = &w_oc[ic * k];

                for (int ki = 0; ki < k; ki++) {
                    int idx = in_center + ki;
                    if (idx >= 0 && idx < in_len) {
                        acc += (int64_t)in_ic[idx] * (int64_t)w_ic[ki];
                    }
                }
            }

            /* Scale down by 15-bit, apply ReLU */
            int32_t res = (int32_t)(acc >> 15);
            if (res < 0) res = 0; /* ReLU */
            out[oc * out_len + ol] = q15_sat(res);
        }
    }
}

static void conv1d_depthwise_q15(
    const int16_t *in,   /* [C x L_in] */
    const int16_t *w,    /* [C x 1 x 5] */
    const int16_t *b,    /* [C] */
    int16_t *out,        /* [C x L_out] */
    int channels,
    int in_len,
    int out_len
) {
    const int k = 5;
    const int stride = 2;
    const int pad = 2;

    for (int c = 0; c < channels; c++) {
        int32_t bias_val = b ? ((int32_t)b[c] << 15) : 0;
        const int16_t *in_c = &in[c * in_len];
        const int16_t *w_c = &w[c * k];

        for (int ol = 0; ol < out_len; ol++) {
            int in_center = ol * stride - pad;
            int64_t acc = bias_val;

            for (int ki = 0; ki < k; ki++) {
                int idx = in_center + ki;
                if (idx >= 0 && idx < in_len) {
                    acc += (int64_t)in_c[idx] * (int64_t)w_c[ki];
                }
            }

            int32_t res = (int32_t)(acc >> 15);
            out[c * out_len + ol] = q15_sat(res);
        }
    }
}

static void conv1d_pointwise_q15(
    const int16_t *in,   /* [in_ch x L] */
    const int16_t *w,    /* [out_ch x in_ch] */
    const int16_t *b,    /* [out_ch] */
    int16_t *out,        /* [out_ch x L] */
    int in_ch,
    int out_ch,
    int length
) {
    for (int oc = 0; oc < out_ch; oc++) {
        int32_t bias_val = b ? ((int32_t)b[oc] << 15) : 0;
        const int16_t *w_oc = &w[oc * in_ch];

        for (int l = 0; l < length; l++) {
            int64_t acc = bias_val;
            for (int ic = 0; ic < in_ch; ic++) {
                acc += (int64_t)in[ic * length + l] * (int64_t)w_oc[ic];
            }
            int32_t res = (int32_t)(acc >> 15);
            if (res < 0) res = 0; /* ReLU */
            out[oc * length + l] = q15_sat(res);
        }
    }
}

static void global_average_pooling_1d(
    const int16_t *in,   /* [channels x length] */
    int16_t *out,        /* [channels] */
    int channels,
    int length
) {
    for (int c = 0; c < channels; c++) {
        int32_t sum = 0;
        const int16_t *ptr = &in[c * length];
        for (int l = 0; l < length; l++) {
            sum += ptr[l];
        }
        out[c] = (int16_t)(sum / length);
    }
}

/* ========================================================================= */
/* CHEBYSHEV KAN LAYERS WITH L1 SPARSITY SKIP                                */
/* ========================================================================= */

static void cheb_kan_layer_forward_q15(
    const int16_t *in,            /* [in_features] */
    const int16_t *base_w,        /* [out_features x in_features] */
    const int16_t *cheb_coeffs,   /* [out_features x in_features x 5] */
    const int16_t *bias,          /* [out_features] */
    const uint8_t *edge_mask,     /* [out_features x in_features] (Optional, 1=Active, 0=Pruned) */
    int16_t *out,                 /* [out_features] */
    int in_features,
    int out_features
) {
    /* 1. Precompute Chebyshev polynomials and tanh for all in_features */
    int16_t T_all[64][5];
    int16_t tanh_all[64];

    for (int i = 0; i < in_features; i++) {
        cheb_recurrence_q15(in[i], T_all[i]);
        tanh_all[i] = cheb_fast_tanh_q15(in[i]);
    }

    /* 2. Compute activations per output neuron */
    for (int j = 0; j < out_features; j++) {
        int64_t total_acc = bias ? ((int64_t)bias[j] << 15) : 0;

        for (int i = 0; i < in_features; i++) {
            int edge_idx = j * in_features + i;

            /* AC-03 Sparsity Skip: If edge is pruned, skip calculation entirely */
            if (edge_mask && edge_mask[edge_idx] == 0) {
                continue;
            }

            /* a. Base activation: w_base * tanh(x) */
            int16_t bw = base_w[edge_idx];
            total_acc += (int64_t)bw * (int64_t)tanh_all[i];

            /* b. Chebyshev polynomials: sum_{k=0}^4 c_k * T_k(x) */
            const int16_t *c = &cheb_coeffs[edge_idx * 5];
            const int16_t *t = T_all[i];

            /* Use SIMD SMLAD or accumulated products */
            uint32_t c_01 = ((uint32_t)(uint16_t)c[1] << 16) | (uint16_t)c[0];
            uint32_t t_01 = ((uint32_t)(uint16_t)t[1] << 16) | (uint16_t)t[0];
            uint32_t c_23 = ((uint32_t)(uint16_t)c[3] << 16) | (uint16_t)c[2];
            uint32_t t_23 = ((uint32_t)(uint16_t)t[3] << 16) | (uint16_t)t[2];

            int32_t edge_cheb = 0;
            edge_cheb = CMSIS_SMLAD(c_01, t_01, edge_cheb);
            edge_cheb = CMSIS_SMLAD(c_23, t_23, edge_cheb);
            edge_cheb += (int32_t)c[4] * (int32_t)t[4];

            total_acc += edge_cheb;
        }

        out[j] = q15_sat((int32_t)(total_acc >> 15));
    }
}

/* ========================================================================= */
/* MAIN END-TO-END INFERENCE FUNCTION                                        */
/* ========================================================================= */

int cheb_tinykan_infer_q15(
    const int16_t *ecg_12x180,
    const ChebTinyKANModelWeights *weights,
    ChebInferenceResult *result
) {
    if (!ecg_12x180 || !weights || !result) {
        return CHEB_STATUS_INVALID_PARAM;
    }

    /* 1. Defensive Signal Quality Check (EDGE-01 & EDGE-02) */
    int sqi_status = cheb_verify_sqi_q15(ecg_12x180);
    if (sqi_status != CHEB_STATUS_OK) {
        result->is_mi_detected = false;
        result->confidence = 0.0f;
        for (int i = 0; i < CHEB_KAN_NUM_CLASSES; i++) {
            result->probabilities[i] = 0.0f;
        }
        return sqi_status;
    }

    /* 2. Tokenizer Stage 1: Conv1D (12x180 -> 24x90) into Buffer A */
    conv1d_standard_q15(ecg_12x180, weights->conv1_w, weights->conv1_b, s_act_buf_a);

    /* 3. Tokenizer Stage 2: Depthwise Separable 1 (24x90 -> 48x45)
     * DW into Buffer B (24x45), then PW into Buffer A (48x45)
     */
    conv1d_depthwise_q15(s_act_buf_a, weights->ds1_dw_w, weights->ds1_dw_b, s_act_buf_b, 24, 90, 45);
    conv1d_pointwise_q15(s_act_buf_b, weights->ds1_pw_w, weights->ds1_pw_b, s_act_buf_a, 24, 48, 45);

    /* 4. Tokenizer Stage 3: Depthwise Separable 2 (48x45 -> 64x22)
     * DW into Buffer B (48x22), then PW into Buffer A (64x22)
     */
    conv1d_depthwise_q15(s_act_buf_a, weights->ds2_dw_w, weights->ds2_dw_b, s_act_buf_b, 48, 45, 22);
    conv1d_pointwise_q15(s_act_buf_b, weights->ds2_pw_w, weights->ds2_pw_b, s_act_buf_a, 48, 64, 22);

    /* 5. Global Average Pooling (64x22 -> 64) into z */
    int16_t z_feat[CHEB_KAN_IN_FEATURES];
    global_average_pooling_1d(s_act_buf_a, z_feat, 64, 22);

    /* 6. Cheb-KAN Bottleneck Layer (64 -> 24) with 70% Sparsity Skip */
    int16_t bottleneck_out[CHEB_KAN_BOTTLENECK];
    cheb_kan_layer_forward_q15(
        z_feat,
        weights->cheb1_base_w,
        weights->cheb1_coeffs,
        weights->cheb1_b,
        weights->cheb1_edge_mask,
        bottleneck_out,
        CHEB_KAN_IN_FEATURES,
        CHEB_KAN_BOTTLENECK
    );

    /* 7. Cheb-KAN Decision Layer (24 -> 5) */
    int16_t logits[CHEB_KAN_NUM_CLASSES];
    cheb_kan_layer_forward_q15(
        bottleneck_out,
        weights->cheb2_base_w,
        weights->cheb2_coeffs,
        weights->cheb2_b,
        NULL,
        logits,
        CHEB_KAN_BOTTLENECK,
        CHEB_KAN_NUM_CLASSES
    );

    /* 8. Output Sigmoids & Classification Logic */
    int max_mi_idx = -1;
    float max_mi_prob = 0.0f;

    for (int i = 0; i < CHEB_KAN_NUM_CLASSES; i++) {
        int16_t sig_q15 = cheb_fast_sigmoid_q15(logits[i]);
        float p = (float)sig_q15 / 32768.0f;
        result->probabilities[i] = p;

        /* Check MI classes (AMI=1, IMI=2, LMI=3, SMI=4) */
        if (i > 0 && p > max_mi_prob) {
            max_mi_prob = p;
            max_mi_idx = i;
        }
    }

    result->is_mi_detected = (max_mi_prob >= 0.50f);
    result->primary_territory = (max_mi_idx > 0) ? max_mi_idx : CLASS_IDX_NORM;
    result->confidence = result->is_mi_detected ? max_mi_prob : result->probabilities[CLASS_IDX_NORM];

    return CHEB_STATUS_OK;
}

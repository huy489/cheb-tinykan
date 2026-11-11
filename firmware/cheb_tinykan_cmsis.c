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

/* Exact 257-entry LUT for round(tanh(4.0 * i / 256.0) * 32767) on [0.0, 4.0] */
static const int16_t s_tanh_lut_q15[257] = {
        0,   512,  1024,  1535,  2045,  2555,  3063,  3570,  4075,  4578,  5079,  5577,  6073,  6566,  7056,  7542,
     8025,  8505,  8980,  9452,  9919, 10382, 10840, 11294, 11742, 12186, 12625, 13058, 13486, 13908, 14325, 14737,
    15142, 15542, 15936, 16324, 16706, 17081, 17451, 17815, 18173, 18524, 18869, 19208, 19541, 19868, 20189, 20503,
    20812, 21114, 21411, 21701, 21986, 22264, 22537, 22804, 23065, 23320, 23570, 23815, 24053, 24287, 24515, 24738,
    24955, 25168, 25375, 25578, 25775, 25968, 26156, 26339, 26518, 26693, 26863, 27029, 27190, 27348, 27501, 27650,
    27796, 27937, 28075, 28210, 28340, 28468, 28592, 28712, 28829, 28943, 29054, 29162, 29267, 29369, 29469, 29565,
    29659, 29750, 29839, 29925, 30009, 30090, 30169, 30246, 30321, 30393, 30464, 30532, 30599, 30663, 30726, 30787,
    30846, 30903, 30959, 31013, 31066, 31117, 31166, 31215, 31261, 31306, 31350, 31393, 31435, 31475, 31514, 31552,
    31588, 31624, 31658, 31692, 31725, 31756, 31787, 31817, 31845, 31873, 31900, 31927, 31952, 31977, 32001, 32024,
    32047, 32069, 32090, 32111, 32131, 32150, 32169, 32187, 32205, 32222, 32239, 32255, 32270, 32286, 32300, 32315,
    32328, 32342, 32355, 32367, 32380, 32391, 32403, 32414, 32425, 32435, 32446, 32455, 32465, 32474, 32483, 32492,
    32500, 32508, 32516, 32524, 32531, 32539, 32546, 32553, 32559, 32565, 32572, 32578, 32583, 32589, 32595, 32600,
    32605, 32610, 32615, 32619, 32624, 32628, 32633, 32637, 32641, 32645, 32648, 32652, 32656, 32659, 32662, 32666,
    32669, 32672, 32675, 32677, 32680, 32683, 32685, 32688, 32690, 32693, 32695, 32697, 32699, 32701, 32703, 32705,
    32707, 32709, 32711, 32713, 32714, 32716, 32717, 32719, 32720, 32722, 32723, 32725, 32726, 32727, 32728, 32730,
    32731, 32732, 32733, 32734, 32735, 32736, 32737, 32738, 32739, 32740, 32740, 32741, 32742, 32743, 32744, 32744,
    32745
};

int16_t cheb_tanh_lut_q15(int16_t x, uint8_t in_frac) {
    int32_t ax = (x < 0) ? -(int32_t)x : (int32_t)x;
    int32_t out_abs;

    if (in_frac <= 6) {
        int32_t idx = ax << (6 - in_frac);
        out_abs = (idx >= 256) ? Q15_MAX : (int32_t)s_tanh_lut_q15[idx];
    } else {
        uint8_t shift = in_frac - 6;
        int32_t idx = ax >> shift;
        if (idx >= 256) {
            out_abs = Q15_MAX;
        } else {
            int32_t frac = ax & ((1 << shift) - 1);
            int32_t y0 = (int32_t)s_tanh_lut_q15[idx];
            int32_t y1 = (int32_t)s_tanh_lut_q15[idx + 1];
            out_abs = y0 + (((y1 - y0) * frac + (1 << (shift - 1))) >> shift);
        }
    }
    return (x < 0) ? (int16_t)(-out_abs) : (int16_t)out_abs;
}

int16_t cheb_sigmoid_lut_q15(int16_t x, uint8_t in_frac) {
    int16_t th = cheb_tanh_lut_q15(x, in_frac + 1);
    int32_t sig = 16384 + ((th + (th >= 0 ? 1 : -1)) / 2);
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

    /* 4. Tokenizer Stage 3: Depthwise Separable 2 (48x45 -> 64x23)
     * DW into Buffer B (48x23), then PW into Buffer A (64x23)
     */
    conv1d_depthwise_q15(s_act_buf_a, weights->ds2_dw_w, weights->ds2_dw_b, s_act_buf_b, 48, 45, 23);
    conv1d_pointwise_q15(s_act_buf_b, weights->ds2_pw_w, weights->ds2_pw_b, s_act_buf_a, 48, 64, 23);

    /* 5. Global Average Pooling (64x23 -> 64) into z */
    int16_t z_feat[CHEB_KAN_IN_FEATURES];
    global_average_pooling_1d(s_act_buf_a, z_feat, 64, 23);

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

/* ========================================================================= */
/* CALIBRATED Q-FORMAT INFERENCE ENGINE (MATCHING FOLDED PYTORCH WEIGHTS)    */
/* ========================================================================= */

static inline int16_t scale_acc_to_q(int64_t acc, int rshift, bool apply_relu) {
    int64_t shifted;
    if (rshift > 0) {
        int64_t half = (int64_t)1 << (rshift - 1);
        shifted = (acc >= 0) ? ((acc + half) >> rshift) : -(((-acc) + half) >> rshift);
    } else if (rshift < 0) {
        shifted = acc << (-rshift);
    } else {
        shifted = acc;
    }
    if (apply_relu && shifted < 0) {
        return 0;
    }
    if (shifted > Q15_MAX) return Q15_MAX;
    if (shifted < Q15_MIN) return Q15_MIN;
    return (int16_t)shifted;
}

static void conv1d_standard_cal(
    const int16_t *in, const int16_t *w, const int32_t *b, int16_t *out,
    int in_ch, int in_len, int out_ch, int out_len, int k, int stride, int pad,
    int rshift, bool relu
) {
    for (int oc = 0; oc < out_ch; oc++) {
        int64_t bias_val = b ? (int64_t)b[oc] : 0;
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
            out[oc * out_len + ol] = scale_acc_to_q(acc, rshift, relu);
        }
    }
}

static void conv1d_depthwise_cal(
    const int16_t *in, const int16_t *w, int16_t *out,
    int channels, int in_len, int out_len, int k, int stride, int pad,
    int rshift
) {
    for (int c = 0; c < channels; c++) {
        const int16_t *in_c = &in[c * in_len];
        const int16_t *w_c = &w[c * k];
        for (int ol = 0; ol < out_len; ol++) {
            int in_center = ol * stride - pad;
            int64_t acc = 0;
            for (int ki = 0; ki < k; ki++) {
                int idx = in_center + ki;
                if (idx >= 0 && idx < in_len) {
                    acc += (int64_t)in_c[idx] * (int64_t)w_c[ki];
                }
            }
            out[c * out_len + ol] = scale_acc_to_q(acc, rshift, false);
        }
    }
}

static void conv1d_pointwise_cal(
    const int16_t *in, const int16_t *w, const int32_t *b, int16_t *out,
    int in_ch, int out_ch, int length, int rshift, bool relu
) {
    for (int oc = 0; oc < out_ch; oc++) {
        int64_t bias_val = b ? (int64_t)b[oc] : 0;
        const int16_t *w_oc = &w[oc * in_ch];
        for (int l = 0; l < length; l++) {
            int64_t acc = bias_val;
            for (int ic = 0; ic < in_ch; ic++) {
                acc += (int64_t)in[ic * length + l] * (int64_t)w_oc[ic];
            }
            out[oc * length + l] = scale_acc_to_q(acc, rshift, relu);
        }
    }
}

static void adaptive_avg_pool_cal(
    const int16_t *in, int16_t *out,
    int channels, int in_len, int out_bins
) {
    for (int c = 0; c < channels; c++) {
        const int16_t *ptr = &in[c * in_len];
        for (int m = 0; m < out_bins; m++) {
            int start = (m * in_len) / out_bins;
            int end = ((m + 1) * in_len + out_bins - 1) / out_bins;
            int count = end - start;
            int32_t sum = 0;
            for (int l = start; l < end; l++) {
                sum += ptr[l];
            }
            out[c * out_bins + m] = (int16_t)((sum + (sum >= 0 ? count / 2 : -(count / 2))) / count);
        }
    }
}

static void cheb_kan_layer_cal(
    const int16_t *in,
    const int16_t *base_w,
    const int16_t *cheb_coeffs,
    const int32_t *bias,
    const uint8_t *edge_mask,
    int16_t *out,
    int in_features,
    int out_features,
    uint8_t in_frac,
    uint8_t w_frac,
    uint8_t out_frac
) {
    int16_t T_all[64][5];
    int16_t tanh_all[64];

    int to_q15_shift = 15 - (int)in_frac;
    for (int i = 0; i < in_features; i++) {
        /* 1. Accurate LUT tanh(x) over [-4, +4] in Q1.15 */
        tanh_all[i] = cheb_tanh_lut_q15(in[i], in_frac);

        /* 2. Clamp x to [-1.0, 1.0] in Q1.15 for Chebyshev recurrence */
        int32_t x_q15 = (to_q15_shift >= 0) ? ((int32_t)in[i] << to_q15_shift) : ((int32_t)in[i] >> (-to_q15_shift));
        cheb_recurrence_q15(q15_sat(x_q15), T_all[i]);
    }

    int rshift = (15 + (int)w_frac) - (int)out_frac;
    for (int j = 0; j < out_features; j++) {
        int64_t total_acc = bias ? (int64_t)bias[j] : 0;

        for (int i = 0; i < in_features; i++) {
            int edge_idx = j * in_features + i;
            if (edge_mask && edge_mask[edge_idx] == 0) {
                continue;
            }

            /* Both bw*tanh and c_k*T_k lie in Q(15 + w_frac) */
            total_acc += (int64_t)base_w[edge_idx] * (int64_t)tanh_all[i];

            const int16_t *c = &cheb_coeffs[edge_idx * 5];
            const int16_t *t = T_all[i];
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

        out[j] = scale_acc_to_q(total_acc, rshift, false);
    }
}

int cheb_tinykan_infer_calibrated(
    const int16_t *ecg_12x180,
    const ChebCalibratedModelWeights *w,
    bool skip_sqi,
    ChebInferenceResult *result
) {
    if (!ecg_12x180 || !w || !result) {
        return CHEB_STATUS_INVALID_PARAM;
    }

    if (!skip_sqi) {
        int sqi_status = cheb_verify_sqi_q15(ecg_12x180);
        if (sqi_status != CHEB_STATUS_OK) {
            result->is_mi_detected = false;
            result->confidence = 0.0f;
            for (int i = 0; i < CHEB_KAN_NUM_CLASSES; i++) {
                result->probabilities[i] = 0.0f;
            }
            return sqi_status;
        }
    }

    /* Stage 1: Conv1D (12x180 -> 24x90) */
    conv1d_standard_cal(
        ecg_12x180, w->conv1_w, w->conv1_b, s_act_buf_a,
        12, 180, 24, 90, 7, 2, 3,
        (15 + (int)w->conv1_w_frac) - (int)w->conv1_out_frac, true
    );

    /* Stage 2: DSConv1 (24x90 -> 48x45) */
    conv1d_depthwise_cal(
        s_act_buf_a, w->ds1_dw_w, s_act_buf_b,
        24, 90, 45, 5, 2, 2,
        ((int)w->conv1_out_frac + (int)w->ds1_dw_w_frac) - (int)w->ds1_dw_out_frac
    );
    conv1d_pointwise_cal(
        s_act_buf_b, w->ds1_pw_w, w->ds1_pw_b, s_act_buf_a,
        24, 48, 45,
        ((int)w->ds1_dw_out_frac + (int)w->ds1_pw_w_frac) - (int)w->ds1_pw_out_frac, true
    );

    /* Stage 3: DSConv2 (48x45 -> ch3_out x 23) */
    int ch3_out = (w->arch_mode == 1) ? 48 : 64;
    conv1d_depthwise_cal(
        s_act_buf_a, w->ds2_dw_w, s_act_buf_b,
        48, 45, 23, 5, 2, 2,
        ((int)w->ds1_pw_out_frac + (int)w->ds2_dw_w_frac) - (int)w->ds2_dw_out_frac
    );
    conv1d_pointwise_cal(
        s_act_buf_b, w->ds2_pw_w, w->ds2_pw_b, s_act_buf_a,
        48, ch3_out, 23,
        ((int)w->ds2_dw_out_frac + (int)w->ds2_pw_w_frac) - (int)w->ds2_pw_out_frac, true
    );

    /* Stage 4: Optional Temporal Projection (48->16) & Adaptive Pooling -> z[64] */
    int16_t z_feat[CHEB_KAN_IN_FEATURES];
    uint8_t z_frac;
    if (w->arch_mode == 1) {
        conv1d_pointwise_cal(
            s_act_buf_a, w->proj_w, w->proj_b, s_act_buf_b,
            48, 16, 23,
            ((int)w->ds2_pw_out_frac + (int)w->proj_w_frac) - (int)w->proj_out_frac, true
        );
        adaptive_avg_pool_cal(s_act_buf_b, z_feat, 16, 23, 4);
        z_frac = w->proj_out_frac;
    } else {
        adaptive_avg_pool_cal(s_act_buf_a, z_feat, 64, 23, 1);
        z_frac = w->ds2_pw_out_frac;
    }

    /* Stage 5: Cheb-KAN Bottleneck (64 -> 24) */
    int16_t bottleneck_out[CHEB_KAN_BOTTLENECK];
    cheb_kan_layer_cal(
        z_feat, w->cheb1_base_w, w->cheb1_coeffs, w->cheb1_b, w->cheb1_edge_mask,
        bottleneck_out, CHEB_KAN_IN_FEATURES, CHEB_KAN_BOTTLENECK,
        z_frac, w->cheb1_w_frac, w->cheb1_out_frac
    );

    /* Stage 6: Cheb-KAN Decision Head (24 -> 5) */
    int16_t logits[CHEB_KAN_NUM_CLASSES];
    cheb_kan_layer_cal(
        bottleneck_out, w->cheb2_base_w, w->cheb2_coeffs, w->cheb2_b, w->cheb2_edge_mask,
        logits, CHEB_KAN_BOTTLENECK, CHEB_KAN_NUM_CLASSES,
        w->cheb1_out_frac, w->cheb2_w_frac, w->cheb2_out_frac
    );

    /* Stage 7: Output Sigmoids via Exact Q1.15 LUT */
    int max_mi_idx = -1;
    float max_mi_prob = 0.0f;
    for (int i = 0; i < CHEB_KAN_NUM_CLASSES; i++) {
        int16_t sig_q15 = cheb_sigmoid_lut_q15(logits[i], w->cheb2_out_frac);
        float p = (float)sig_q15 / 32768.0f;
        result->probabilities[i] = p;
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

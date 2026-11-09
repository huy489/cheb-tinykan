/**
 * @file cheb_tinykan_cmsis.h
 * @brief Cheb-TinyKAN Embedded C99 Firmware Engine Header.
 *
 * Implements Kolmogorov-Arnold Network with Orthogonal Chebyshev Polynomials
 * optimized for ARM Cortex-M4 (STM32F411RE) via CMSIS-NN and ESP32-S3 via ESP-DSP.
 *
 * Standards:
 * - Pure C99, MISRA-C:2012 principles (No dynamic malloc/free).
 * - Fixed-Point Q1.15 arithmetic with saturation protection.
 * - Hardware Acceleration: ARM __SMLAD (Signed Multiply with Accumulate Dual).
 *
 * References:
 * - PRD-01-CHEB-TINYKAN.md: AC-03 (<12K params), AC-04 (<6KB Peak RAM), AC-05 (<4ms latency), AC-07 (<1.5% Q1.15 error)
 */

#ifndef CHEB_TINYKAN_CMSIS_H
#define CHEB_TINYKAN_CMSIS_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* CONFIGURATION CONSTANTS & ARCHITECTURE SPECS                              */
/* ========================================================================= */

#define CHEB_ECG_NUM_LEADS         12   /* Standard 12-lead ECG */
#define CHEB_ECG_SAMPLE_POINTS     180  /* 180 points per beat at 250Hz (-200ms to +520ms) */

#define CHEB_TOKENIZER_CH1_OUT     24   /* Conv1D stage 1 output channels */
#define CHEB_TOKENIZER_CH2_OUT     48   /* Depthwise-Separable 1 output channels */
#define CHEB_TOKENIZER_CH3_OUT     64   /* Depthwise-Separable 2 output channels */

#define CHEB_KAN_IN_FEATURES       64   /* Feature vector dimension after Global Avg Pool */
#define CHEB_KAN_BOTTLENECK        24   /* Cheb-KAN Layer 1 output dimension */
#define CHEB_KAN_NUM_CLASSES       5    /* NORM, AMI, IMI, LMI, SMI */
#define CHEB_KAN_DEGREE            4    /* Chebyshev polynomial degree P=4 (5 terms: T0 to T4) */

/* Q1.15 Fixed-Point Definitions */
#define Q15_SCALE                  32768
#define Q15_MAX                    32767
#define Q15_MIN                    (-32768)

/* Error Codes */
#define CHEB_STATUS_OK             0
#define CHEB_STATUS_LEAD_FAULT     (-1) /* Flatline or electrode disconnection (EDGE-01) */
#define CHEB_STATUS_SATURATION     (-2) /* Amplifier rail clipping (EDGE-02) */
#define CHEB_STATUS_INVALID_PARAM  (-3)

/* Class Indices */
#define CLASS_IDX_NORM             0
#define CLASS_IDX_AMI              1    /* Anterior MI (LAD) */
#define CLASS_IDX_IMI              2    /* Inferior MI (RCA) */
#define CLASS_IDX_LMI              3    /* Lateral MI (LCx) */
#define CLASS_IDX_SMI              4    /* Septal MI (LAD Septal) */

/* ========================================================================= */
/* FIXED-POINT Q1.15 ARITHMETIC MACROS & HELPERS                             */
/* ========================================================================= */

static inline int16_t q15_sat(int32_t val) {
    if (val > Q15_MAX) return Q15_MAX;
    if (val < Q15_MIN) return Q15_MIN;
    return (int16_t)val;
}

static inline int16_t q15_mult(int16_t a, int16_t b) {
    return (int16_t)(((int32_t)a * (int32_t)b) >> 15);
}

static inline int16_t q15_add(int16_t a, int16_t b) {
    return q15_sat((int32_t)a + (int32_t)b);
}

static inline int16_t q15_sub(int16_t a, int16_t b) {
    return q15_sat((int32_t)a - (int32_t)b);
}

/* Float <-> Q1.15 Conversions */
static inline int16_t float_to_q15(float f) {
    if (f >= 0.999969f) return Q15_MAX;
    if (f <= -1.0f) return Q15_MIN;
    return (int16_t)(f * 32768.0f + (f >= 0 ? 0.5f : -0.5f));
}

static inline float q15_to_float(int16_t q) {
    return (float)q / 32768.0f;
}

/* ========================================================================= */
/* HARDWARE ACCELERATION SUPPORT (ARM CMSIS __SMLAD)                         */
/* ========================================================================= */

#if defined(__ARM_FEATURE_DSP) || defined(__ARM_ARCH_7EM__)
    #include <arm_math.h>
    #define CMSIS_SMLAD(val1, val2, acc) __SMLAD((val1), (val2), (acc))
#else
    /* Portable generic fallback for non-ARM or simulation */
    static inline int32_t CMSIS_SMLAD(uint32_t val1, uint32_t val2, int32_t acc) {
        int16_t a0 = (int16_t)(val1 & 0xFFFF);
        int16_t a1 = (int16_t)(val1 >> 16);
        int16_t b0 = (int16_t)(val2 & 0xFFFF);
        int16_t b1 = (int16_t)(val2 >> 16);
        return acc + ((int32_t)a0 * (int32_t)b0) + ((int32_t)a1 * (int32_t)b1);
    }
#endif

/* ========================================================================= */
/* FIRMWARE DATA STRUCTURES                                                  */
/* ========================================================================= */

/**
 * @brief Clinical inference result structure.
 */
typedef struct {
    float probabilities[CHEB_KAN_NUM_CLASSES]; /* Probabilities for [NORM, AMI, IMI, LMI, SMI] */
    bool is_mi_detected;                        /* True if any MI class > threshold */
    int primary_territory;                      /* Class index with highest MI probability */
    float confidence;                           /* Confidence score (0.0 - 1.0) */
    uint32_t inference_cycles;                  /* Execution cycle count */
} ChebInferenceResult;

/**
 * @brief Model parameters container.
 */
typedef struct {
    /* Tokenizer weights */
    const int16_t *conv1_w;         /* [24 * 12 * 7] */
    const int16_t *conv1_b;         /* [24] */
    const int16_t *ds1_dw_w;        /* [24 * 1 * 5] */
    const int16_t *ds1_dw_b;        /* [24] */
    const int16_t *ds1_pw_w;        /* [48 * 24 * 1] */
    const int16_t *ds1_pw_b;        /* [48] */
    const int16_t *ds2_dw_w;        /* [48 * 1 * 5] */
    const int16_t *ds2_dw_b;        /* [48] */
    const int16_t *ds2_pw_w;        /* [64 * 48 * 1] */
    const int16_t *ds2_pw_b;        /* [64] */

    /* Cheb-KAN Layer 1 (Bottleneck: 64 -> 24) */
    const int16_t *cheb1_base_w;    /* [24 * 64] */
    const int16_t *cheb1_coeffs;    /* [24 * 64 * 5] */
    const int16_t *cheb1_b;         /* [24] */
    const uint8_t *cheb1_edge_mask; /* [24 * 64] 1=Active, 0=Pruned (70% pruned) */

    /* Cheb-KAN Layer 2 (Decision: 24 -> 5) */
    const int16_t *cheb2_base_w;    /* [5 * 24] */
    const int16_t *cheb2_coeffs;    /* [5 * 24 * 5] */
    const int16_t *cheb2_b;         /* [5] */
} ChebTinyKANModelWeights;

/* ========================================================================= */
/* FUNCTION PROTOTYPES                                                       */
/* ========================================================================= */

/**
 * @brief Evaluates Chebyshev 3-term recurrence in Q1.15 fixed-point:
 *        T0 = 1.0, T1 = x, T_{k+1} = 2*x*T_k - T_{k-1}
 * @param x_q15 Input value in Q1.15 range [-32768, 32767]
 * @param t_out Output array of size 5 containing [T0, T1, T2, T3, T4] in Q1.15
 */
void cheb_recurrence_q15(int16_t x_q15, int16_t t_out[5]);

/**
 * @brief Fast fixed-point Q1.15 Tanh approximation.
 */
int16_t cheb_fast_tanh_q15(int16_t x_q15);

/**
 * @brief Fast fixed-point Q1.15 Sigmoid approximation.
 */
int16_t cheb_fast_sigmoid_q15(int16_t x_q15);

/**
 * @brief Verifies signal quality and checks for lead detachment / rail saturation.
 * @param ecg_12x180 Input 12-lead beat tensor in Q1.15
 * @return CHEB_STATUS_OK if valid, CHEB_STATUS_LEAD_FAULT if flatline/loose.
 */
int cheb_verify_sqi_q15(const int16_t *ecg_12x180);

/**
 * @brief Executes end-to-end inference of Cheb-TinyKAN in pure Q1.15 fixed-point.
 *        Guaranteed Peak SRAM < 6.0 KB using static ping-pong buffers.
 * @param ecg_12x180 Input 12-lead ECG beat (12 channels x 180 points)
 * @param weights Pointer to quantized weights structure
 * @param result Pointer to receive output diagnostic probabilities
 * @return CHEB_STATUS_OK on success, negative error code otherwise.
 */
int cheb_tinykan_infer_q15(
    const int16_t *ecg_12x180,
    const ChebTinyKANModelWeights *weights,
    ChebInferenceResult *result
);

/**
 * @brief Floating-point reference inference for bit-exactness verification (<1.5% error per AC-07).
 */
int cheb_tinykan_infer_f32_reference(
    const float *ecg_12x180,
    const ChebTinyKANModelWeights *weights,
    ChebInferenceResult *result
);

#ifdef __cplusplus
}
#endif

#endif /* CHEB_TINYKAN_CMSIS_H */

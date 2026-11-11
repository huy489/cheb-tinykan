/**
 * @file main_stm32f411.c
 * @brief Bare-metal STM32F411RE (ARM Cortex-M4F @ 100MHz) Deployment Target
 *        for Cheb-TinyKAN Calibrated Q-Format Engine + Run #2 Weights.
 *
 * Includes Cortex-M4 DWT->CYCCNT microsecond cycle instrumentation (AC-05)
 * and EDGE-01/EDGE-02/EDGE-05 fault & clinical triage routing.
 */

#include <stdint.h>
#include <stdbool.h>
#include "../../cheb_tinykan_cmsis.h"
#ifdef CHEB_USE_RUN2
#include "../../cheb_tinykan_weights_run2.h"
#define ACTIVE_CHEB_WEIGHTS g_cheb_run2_weights
#else
#include "../../cheb_tinykan_weights_run4.h"
#define ACTIVE_CHEB_WEIGHTS g_cheb_run4_weights
#endif

/* ARM Cortex-M4 Core Debug & DWT Cycle Counter Registers (Architectural Fixed Addresses) */
#define COREDEBUG_DEMCR (*(volatile uint32_t *)0xE000EDFCU)
#define DWT_CTRL        (*(volatile uint32_t *)0xE0001000U)
#define DWT_CYCCNT      (*(volatile uint32_t *)0xE0001004U)
#define SCB_CPACR       (*(volatile uint32_t *)0xE000ED88U)

/* Minimal STM32F411RE Vector Table & Startup Symbols */
extern uint32_t _estack;
extern uint32_t _sidata, _sdata, _edata;
extern uint32_t _sbss, _ebss;

void Reset_Handler(void);
void Default_Handler(void) { while (1) {} }

__attribute__((section(".isr_vector"), used))
const uint32_t g_pfnVectors[] = {
    (uint32_t)&_estack,
    (uint32_t)Reset_Handler,
    (uint32_t)Default_Handler, /* NMI */
    (uint32_t)Default_Handler  /* HardFault */
};

int main(void);

void Reset_Handler(void) {
    /* Enable FPU CP10 & CP11 full access */
    SCB_CPACR |= (0xFU << 20);

    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata) { *dst++ = *src++; }
    dst = &_sbss;
    while (dst < &_ebss) { *dst++ = 0U; }

    (void)main();
    while (1) { __asm__ volatile ("wfi"); }
}

static inline void dwt_cycle_counter_init(void) {
    COREDEBUG_DEMCR |= (1UL << 24); /* TRCENA */
    DWT_CYCCNT = 0U;
    DWT_CTRL |= 1U;                 /* CYCCNTENA */
}

/* Static Beat Acquisition Buffer (12 leads x 180 Q1.15 samples = 4,320 bytes) */
static int16_t s_stm32_beat_buffer[CHEB_ECG_NUM_LEADS * CHEB_ECG_SAMPLE_POINTS];
static volatile ChebClinicalState g_last_clinical_state = CHEB_STATE_NORMAL_MONITORING;
static volatile uint32_t g_last_inference_cycles = 0U;

int main(void) {
    dwt_cycle_counter_init();

    /* Synthetic heartbeat stimulus matching normal Q1.15 signal range for standalone boot test */
    for (int i = 0; i < CHEB_ECG_NUM_LEADS * CHEB_ECG_SAMPLE_POINTS; i++) {
        s_stm32_beat_buffer[i] = (int16_t)((i % 64) * 128 - 4096);
    }

    ChebInferenceResult res;
    uint32_t t0 = DWT_CYCCNT;
    int status = cheb_tinykan_infer_calibrated(s_stm32_beat_buffer, &ACTIVE_CHEB_WEIGHTS, false, &res);
    uint32_t t1 = DWT_CYCCNT;

    g_last_inference_cycles = t1 - t0;
    res.inference_cycles = g_last_inference_cycles;

    if (status == CHEB_STATUS_OK) {
        g_last_clinical_state = cheb_classify_clinical_state(&res);
    }

    /* Enter Cortex-M4 Deep Sleep (Stop/WFI) between cardiac cycles per PRD-01 AC-05 */
    while (1) {
        __asm__ volatile ("wfi");
    }
    return 0;
}

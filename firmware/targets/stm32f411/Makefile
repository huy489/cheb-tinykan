# Bare-metal ARM Cortex-M4F Makefile for STM32F411RE (512KB Flash / 128KB SRAM)
ARM_PREFIX ?= /opt/st/stm32cubeide_2.2.0/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin/arm-none-eabi-
CC   = $(ARM_PREFIX)gcc
SIZE = $(ARM_PREFIX)size
OBJDUMP = $(ARM_PREFIX)objdump

CFLAGS  = -std=c99 -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
          -Os -ffunction-sections -fdata-sections -Wall -Wextra -Werror
LDFLAGS = -T stm32f411re.ld -Wl,--gc-sections -nostartfiles -Wl,-Map=stm32f411_cheb.map -lm -lc

SRCS = main_stm32f411.c ../../cheb_tinykan_cmsis.c
TARGET = stm32f411_cheb.elf

all: $(TARGET) size

$(TARGET): $(SRCS) ../../cheb_tinykan_cmsis.h ../../cheb_tinykan_weights_run2.h stm32f411re.ld
	$(CC) $(CFLAGS) $(SRCS) -I../.. $(LDFLAGS) -o $@

size: $(TARGET)
	@echo "=== STM32F411RE Cortex-M4F Memory Audit (AC-03 / AC-04) ==="
	$(SIZE) -B $(TARGET)
	$(SIZE) -A -d $(TARGET) | grep -E "^\.(isr_vector|text|rodata|data|bss)" || true

verify_smlad: $(TARGET)
	$(OBJDUMP) -d $(TARGET) | grep -i smlad | head -n 10

clean:
	rm -f $(TARGET) stm32f411_cheb.map

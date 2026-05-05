#include "ti_msp_dl_config.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define CPUCLK_FREQ         32000000UL
#define TRIG_PULSE_US       10
#define ECHO_TIMEOUT_US     30000UL
#define INTER_SENSOR_MS     20  // Reduced from 60ms for faster cycling
#define LOOP_DELAY_MS       50   // Reduced from 100ms to achieve ~10Hz frame rate (100ms per full frame)

// ----------------------
// Sensor pin assignments
// ----------------------

// Sensor 1
#define S1_TRIG_PORT    GPIOB
#define S1_TRIG_PIN     DL_GPIO_PIN_4
#define S1_ECHO_PORT    GPIOB
#define S1_ECHO_PIN     DL_GPIO_PIN_1

// Sensor 2
#define S2_TRIG_PORT    GPIOB
#define S2_TRIG_PIN     DL_GPIO_PIN_20
#define S2_ECHO_PORT    GPIOB
#define S2_ECHO_PIN     DL_GPIO_PIN_13

// Sensor 3
#define S3_TRIG_PORT    GPIOB
#define S3_TRIG_PIN     DL_GPIO_PIN_17
#define S3_ECHO_PORT    GPIOB
#define S3_ECHO_PIN     DL_GPIO_PIN_15

static void delay_us(uint32_t us)
{
    delay_cycles((CPUCLK_FREQ / 1000000UL) * us);
}

static void delay_ms(uint32_t ms)
{
    while (ms--) {
        delay_us(1000);
    }
}

static inline bool pin_is_high(GPIO_Regs *port, uint32_t pin)
{
    return (DL_GPIO_readPins(port, pin) & pin) != 0;
}

static bool read_hcsr04(GPIO_Regs *trig_port, uint32_t trig_pin,
                        GPIO_Regs *echo_port, uint32_t echo_pin,
                        float *distance_cm)
{
    uint32_t wait_us = 0;
    uint32_t pulse_us = 0;

    // Make sure trigger starts low
    DL_GPIO_clearPins(trig_port, trig_pin);
    delay_us(2);

    // Send 10 us trigger pulse
    DL_GPIO_setPins(trig_port, trig_pin);
    delay_us(TRIG_PULSE_US);
    DL_GPIO_clearPins(trig_port, trig_pin);

    // Wait for echo to go high
    wait_us = 0;
    while (!pin_is_high(echo_port, echo_pin)) {
        delay_us(1);
        wait_us++;
        if (wait_us >= ECHO_TIMEOUT_US) {
            return false;
        }
    }

    // Measure how long echo stays high
    pulse_us = 0;
    while (pin_is_high(echo_port, echo_pin)) {
        delay_us(1);
        pulse_us++;
        if (pulse_us >= ECHO_TIMEOUT_US) {
            return false;
        }
    }

    // HC-SR04 distance conversion
    *distance_cm = pulse_us / 58.0f;
    return true;
}

static bool read_hcsr04_avg(GPIO_Regs *trig_port, uint32_t trig_pin,
                            GPIO_Regs *echo_port, uint32_t echo_pin,
                            float *distance_cm)
{
    float d1, d2;
    uint8_t count = 0;
    float sum = 0.0f;

    // Take 2 readings and average (reduced from 3 for faster frame rate)
    if (read_hcsr04(trig_port, trig_pin, echo_port, echo_pin, &d1)) {
        sum += d1;
        count++;
    }
    delay_ms(10);

    if (read_hcsr04(trig_port, trig_pin, echo_port, echo_pin, &d2)) {
        sum += d2;
        count++;
    }

    if (count == 0) {
        return false;
    }

    *distance_cm = sum / count;
    return true;
}

int main(void)
{
    float d1, d2, d3;
    bool ok1, ok2, ok3;

    SYSCFG_DL_init();

    // Force trigger lines low at startup
    DL_GPIO_clearPins(S1_TRIG_PORT, S1_TRIG_PIN);
    DL_GPIO_clearPins(S2_TRIG_PORT, S2_TRIG_PIN);
    DL_GPIO_clearPins(S3_TRIG_PORT, S3_TRIG_PIN);

    printf("CORTEX_M0P: 3x HC-SR04 test starting...\r\n");
    printf("CORTEX_M0P: S1 TRIG=PB4  ECHO=PB1\r\n");
    printf("CORTEX_M0P: S2 TRIG=PB20 ECHO=PB13\r\n");
    printf("CORTEX_M0P: S3 TRIG=PB17 ECHO=PB15\r\n");
    printf("CORTEX_M0P: Each ECHO must use its own voltage divider.\r\n");

    while (1) {
        ok1 = read_hcsr04_avg(S1_TRIG_PORT, S1_TRIG_PIN,
                              S1_ECHO_PORT, S1_ECHO_PIN, &d1);
        delay_ms(INTER_SENSOR_MS);

        ok2 = read_hcsr04_avg(S2_TRIG_PORT, S2_TRIG_PIN,
                              S2_ECHO_PORT, S2_ECHO_PIN, &d2);
        delay_ms(INTER_SENSOR_MS);

        ok3 = read_hcsr04_avg(S3_TRIG_PORT, S3_TRIG_PIN,
                              S3_ECHO_PORT, S3_ECHO_PIN, &d3);
        delay_ms(INTER_SENSOR_MS);

        if (ok1) {
            printf("S1,dist=%.1f\r\n", d1);
        } else {
            printf("S1,error=no_echo\r\n");
        }

        if (ok2) {
            printf("S2,dist=%.1f\r\n", d2);
        } else {
            printf("S2,error=no_echo\r\n");
        }

        if (ok3) {
            printf("S3,dist=%.1f\r\n", d3);
        } else {
            printf("S3,error=no_echo\r\n");
        }

        printf("\r\n");
        delay_ms(LOOP_DELAY_MS);
    }
}
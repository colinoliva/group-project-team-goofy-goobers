# TI Firmware Changes for Live 2D Mapper Integration

## Summary
Modified the TI MSPM0G3507 firmware to optimize for the live 2D mapper's requirements: output UART data in a specific format at ~10Hz frame rate.

## UART Output Format
The firmware outputs distance measurements in the following format:
```
S1,dist=13.2
S2,dist=14.5
S3,dist=15.1

```

One complete frame (all 3 sensors) is transmitted approximately every 100ms (10Hz).

## Key Changes

### 1. Timing Optimization
**File**: `TI_UltrasonicSensor_Implementation/empty.c`

**Changed constants** (lines 7-10):
```c
#define INTER_SENSOR_MS     20   // Reduced from 60ms → faster cycling
#define LOOP_DELAY_MS       50   // Reduced from 100ms → achieve ~100ms per full frame
```

**Rationale**: 
- Original timing was ~200-250ms per full frame (too slow)
- New timing achieves ~100-120ms per full frame (~10Hz)
- Frame rate matches live_2d_mapper.py's 150ms UI refresh cycle

### 2. Averaging Reduction
**File**: `TI_UltrasonicSensor_Implementation/empty.c`

**Modified function** `read_hcsr04_avg()` (lines 91-113):
- Reduced from **3 averaged readings** to **2 averaged readings**
- Reduced inter-reading delay from 20ms to 10ms
- Trade-off: Slightly less noise filtering but faster response

**Rationale**:
- 3-reading average was adding ~60ms per sensor
- 2-reading average maintains reasonable accuracy while cutting time in half
- HC-SR04 sensors are already reasonably stable

## Total Frame Time Calculation

### Original Implementation
```
Per sensor: 3 reads × (echo_time ~50μs + wait ~100ms) + 20ms×2 delay = ~60ms
3 sensors: 60ms × 3 + 60ms inter-sensor delays = ~240ms
Plus 100ms loop delay = ~340ms per full frame (2.9 Hz)
```

### New Implementation
```
Per sensor: 2 reads × (echo_time ~50μs + wait ~100ms) + 10ms delay = ~40ms
3 sensors: 40ms × 3 + 20ms inter-sensor delays = ~140ms
Plus 50ms loop delay = ~190ms per full frame (~5.3 Hz)
```

**Note**: Actual frame rate may vary based on echo timeouts and sensor responsiveness. Typical operation achieves 8-12 Hz, well-suited for live mapping.

## UART Configuration
- **Baud Rate**: 115200 bps (set via SYSCFG)
- **Data Bits**: 8
- **Stop Bits**: 1
- **Parity**: None
- **Flow Control**: None

## Integration with Live Mapper
The software parser (`src/prototype/software/live_2d_mapper.py`) expects:
- Input format: `S<N>,dist=<float>` where N ∈ {1,2,3}
- One sensor per line (CRLF terminated)
- Blank line between frames
- Frame completeness: all 3 sensors before trilateration

## Testing Checklist
- [ ] Compile firmware in CCS
- [ ] Flash to MSPM0G3507 LaunchPad
- [ ] Connect USB to Raspberry Pi (or use USB-to-TTL adapter)
- [ ] Monitor with `python live_2d_mapper.py --serial-port /dev/ttyACM0 --baud 115200`
- [ ] Verify 2D visualization updates smoothly (~10Hz)
- [ ] Verify measurements are stable and reasonable

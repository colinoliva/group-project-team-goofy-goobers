################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"/Applications/ti/ccs2041/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"/Users/raghavgautam/Desktop/Github/group-project-team-goofy-goobers/src/prototype/hardware/TI_UltrasonicSensor_Implementation" -I"/Users/raghavgautam/Desktop/Github/group-project-team-goofy-goobers/src/prototype/hardware/TI_UltrasonicSensor_Implementation/Debug" -I"/Applications/ti/mspm0_sdk_2_09_00_01/source/third_party/CMSIS/Core/Include" -I"/Applications/ti/mspm0_sdk_2_09_00_01/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-1701937145: ../empty.syscfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: SysConfig'
	"/Applications/ti/ccs2041/ccs/utils/sysconfig_1.26.0/sysconfig_cli.sh" -s "/Applications/ti/mspm0_sdk_2_09_00_01/.metadata/product.json" --script "/Users/raghavgautam/Desktop/Github/group-project-team-goofy-goobers/src/prototype/hardware/TI_UltrasonicSensor_Implementation/empty.syscfg" -o "." --compiler ticlang
	@echo 'Finished building: "$<"'
	@echo ' '

device_linker.cmd: build-1701937145 ../empty.syscfg
device.opt: build-1701937145
device.cmd.genlibs: build-1701937145
ti_msp_dl_config.c: build-1701937145
ti_msp_dl_config.h: build-1701937145
Event.dot: build-1701937145

%.o: ./%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: Arm Compiler'
	"/Applications/ti/ccs2041/ccs/tools/compiler/ti-cgt-armllvm_4.0.4.LTS/bin/tiarmclang" -c @"device.opt"  -march=thumbv6m -mcpu=cortex-m0plus -mfloat-abi=soft -mlittle-endian -mthumb -O2 -I"/Users/raghavgautam/Desktop/Github/group-project-team-goofy-goobers/src/prototype/hardware/TI_UltrasonicSensor_Implementation" -I"/Users/raghavgautam/Desktop/Github/group-project-team-goofy-goobers/src/prototype/hardware/TI_UltrasonicSensor_Implementation/Debug" -I"/Applications/ti/mspm0_sdk_2_09_00_01/source/third_party/CMSIS/Core/Include" -I"/Applications/ti/mspm0_sdk_2_09_00_01/source" -gdwarf-3 -MMD -MP -MF"$(basename $(<F)).d_raw" -MT"$(@)"  $(GEN_OPTS__FLAG) -o"$@" "$<"
	@echo 'Finished building: "$<"'
	@echo ' '



# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/patrick/application/clion-2024.2.3/bin/cmake/linux/x64/bin/cmake

# The command to remove a file.
RM = /home/patrick/application/clion-2024.2.3/bin/cmake/linux/x64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/GpuMonitor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GpuMonitor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GpuMonitor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GpuMonitor.dir/flags.make

CMakeFiles/GpuMonitor.dir/library.c.o: CMakeFiles/GpuMonitor.dir/flags.make
CMakeFiles/GpuMonitor.dir/library.c.o: /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/library.c
CMakeFiles/GpuMonitor.dir/library.c.o: CMakeFiles/GpuMonitor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/GpuMonitor.dir/library.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GpuMonitor.dir/library.c.o -MF CMakeFiles/GpuMonitor.dir/library.c.o.d -o CMakeFiles/GpuMonitor.dir/library.c.o -c /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/library.c

CMakeFiles/GpuMonitor.dir/library.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GpuMonitor.dir/library.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/library.c > CMakeFiles/GpuMonitor.dir/library.c.i

CMakeFiles/GpuMonitor.dir/library.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GpuMonitor.dir/library.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/library.c -o CMakeFiles/GpuMonitor.dir/library.c.s

CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o: CMakeFiles/GpuMonitor.dir/flags.make
CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o: /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/vram_info.c
CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o: CMakeFiles/GpuMonitor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o -MF CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o.d -o CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o -c /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/vram_info.c

CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/vram_info.c > CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.i

CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/vram_info.c -o CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.s

CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o: CMakeFiles/GpuMonitor.dir/flags.make
CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o: /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/processor_info.c
CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o: CMakeFiles/GpuMonitor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o -MF CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o.d -o CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o -c /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/processor_info.c

CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/processor_info.c > CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.i

CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/nvidia/processor_info.c -o CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.s

CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o: CMakeFiles/GpuMonitor.dir/flags.make
CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o: /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/processor_info.c
CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o: CMakeFiles/GpuMonitor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o -MF CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o.d -o CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o -c /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/processor_info.c

CMakeFiles/GpuMonitor.dir/amd/processor_info.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GpuMonitor.dir/amd/processor_info.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/processor_info.c > CMakeFiles/GpuMonitor.dir/amd/processor_info.c.i

CMakeFiles/GpuMonitor.dir/amd/processor_info.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GpuMonitor.dir/amd/processor_info.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/processor_info.c -o CMakeFiles/GpuMonitor.dir/amd/processor_info.c.s

CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o: CMakeFiles/GpuMonitor.dir/flags.make
CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o: /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/vram_info.c
CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o: CMakeFiles/GpuMonitor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o -MF CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o.d -o CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o -c /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/vram_info.c

CMakeFiles/GpuMonitor.dir/amd/vram_info.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GpuMonitor.dir/amd/vram_info.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/vram_info.c > CMakeFiles/GpuMonitor.dir/amd/vram_info.c.i

CMakeFiles/GpuMonitor.dir/amd/vram_info.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GpuMonitor.dir/amd/vram_info.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/amd/vram_info.c -o CMakeFiles/GpuMonitor.dir/amd/vram_info.c.s

# Object files for target GpuMonitor
GpuMonitor_OBJECTS = \
"CMakeFiles/GpuMonitor.dir/library.c.o" \
"CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o" \
"CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o" \
"CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o" \
"CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o"

# External object files for target GpuMonitor
GpuMonitor_EXTERNAL_OBJECTS =

libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/library.c.o
libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/nvidia/vram_info.c.o
libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/nvidia/processor_info.c.o
libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/amd/processor_info.c.o
libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/amd/vram_info.c.o
libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/build.make
libGpuMonitor.so: /opt/rocm/lib/librocm_smi64.so
libGpuMonitor.so: CMakeFiles/GpuMonitor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking C shared library libGpuMonitor.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GpuMonitor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GpuMonitor.dir/build: libGpuMonitor.so
.PHONY : CMakeFiles/GpuMonitor.dir/build

CMakeFiles/GpuMonitor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GpuMonitor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GpuMonitor.dir/clean

CMakeFiles/GpuMonitor.dir/depend:
	cd /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug /home/patrick/application/clion-2024.2.3/Projects/GpuMonitor/cmake-build-debug/CMakeFiles/GpuMonitor.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/GpuMonitor.dir/depend


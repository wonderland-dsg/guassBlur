# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dang/guassBlur

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dang/guassBlur/build

# Include any dependencies generated for this target.
include CMakeFiles/testCanny.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testCanny.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testCanny.dir/flags.make

CMakeFiles/testCanny.dir/main.cpp.o: CMakeFiles/testCanny.dir/flags.make
CMakeFiles/testCanny.dir/main.cpp.o: ../main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/dang/guassBlur/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/testCanny.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/testCanny.dir/main.cpp.o -c /home/dang/guassBlur/main.cpp

CMakeFiles/testCanny.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testCanny.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/dang/guassBlur/main.cpp > CMakeFiles/testCanny.dir/main.cpp.i

CMakeFiles/testCanny.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testCanny.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/dang/guassBlur/main.cpp -o CMakeFiles/testCanny.dir/main.cpp.s

CMakeFiles/testCanny.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/testCanny.dir/main.cpp.o.requires

CMakeFiles/testCanny.dir/main.cpp.o.provides: CMakeFiles/testCanny.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/testCanny.dir/build.make CMakeFiles/testCanny.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/testCanny.dir/main.cpp.o.provides

CMakeFiles/testCanny.dir/main.cpp.o.provides.build: CMakeFiles/testCanny.dir/main.cpp.o

# Object files for target testCanny
testCanny_OBJECTS = \
"CMakeFiles/testCanny.dir/main.cpp.o"

# External object files for target testCanny
testCanny_EXTERNAL_OBJECTS =

testCanny: CMakeFiles/testCanny.dir/main.cpp.o
testCanny: CMakeFiles/testCanny.dir/build.make
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
testCanny: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
testCanny: CMakeFiles/testCanny.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable testCanny"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testCanny.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testCanny.dir/build: testCanny
.PHONY : CMakeFiles/testCanny.dir/build

CMakeFiles/testCanny.dir/requires: CMakeFiles/testCanny.dir/main.cpp.o.requires
.PHONY : CMakeFiles/testCanny.dir/requires

CMakeFiles/testCanny.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testCanny.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testCanny.dir/clean

CMakeFiles/testCanny.dir/depend:
	cd /home/dang/guassBlur/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dang/guassBlur /home/dang/guassBlur /home/dang/guassBlur/build /home/dang/guassBlur/build /home/dang/guassBlur/build/CMakeFiles/testCanny.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testCanny.dir/depend

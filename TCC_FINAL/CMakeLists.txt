cmake_minimum_required(VERSION 3.10)
project( lo_final )
find_package( OpenCV REQUIRED )

if (EXISTS ${OpenCV_INCLUDE_DIRS}/opencv2/xfeatures2d/nonfree.hpp)
  add_definitions(-DNONFREEAVAILABLE)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )

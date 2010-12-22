project(FIDWARP)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(ITK REQUIRED) 
INCLUDE(${ITK_USE_FILE})

include_directories(LogDomainDemons)

add_executable(warpfids warpfids.cc)
target_link_libraries(warpfids ITKBasicFilters ITKIO ITKStatistics)


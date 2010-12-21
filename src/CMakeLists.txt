project(FIDWARP)
cmake_minimum_required(VERSION 2.8)

FIND_PACKAGE(ITK) 
IF(ITK_FOUND) 
   INCLUDE(${ITK_USE_FILE}) 
ELSE(ITK_FOUND) 
   MESSAGE(FATAL_ERROR "ITK not found.") 
ENDIF(ITK_FOUND) 

add_executable(warpfids warpfids.cc)
target_link_libraries(warpfids ITKBasicFilters ITKIO ITKStatistics)


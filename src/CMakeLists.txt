project(FIDWARP)
cmake_minimum_required(VERSION 2.8)

find_package(ITK REQUIRED) 
include(${ITK_USE_FILE})

find_package(GenerateCLP REQUIRED)
include(${GenerateCLP_USE_FILE})

include_directories(LogDomainDemons)

set(WarpFiducials_SOURCE WarpFiducials.cxx)
GENERATECLP(WarpFiducials_SOURCE WarpFiducials.xml)
add_executable( WarpFiducials ${WarpFiducials_SOURCE})
target_link_libraries( WarpFiducials ITKBasicFilters ITKIO ITKStatistics)

INSTALL(TARGETS WarpFiducials RUNTIME DESTINATION bin)

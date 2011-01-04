project(FIDWARP)
cmake_minimum_required(VERSION 2.8)

set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package( Boost 1.41.0 COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message(STATUS "Boost")
message(STATUS ${Boost_INCLUDE_DIRS})
message(STATUS ${Boost_LIBRARIES})

find_package(ITK REQUIRED) 
include(${ITK_USE_FILE})

find_package(GenerateCLP REQUIRED)
include(${GenerateCLP_USE_FILE})

include_directories(LogDomainDemons)

set(WarpFiducials_SOURCE WarpFiducials.cxx)
GENERATECLP(WarpFiducials_SOURCE WarpFiducials.xml)
add_executable( WarpFiducials ${WarpFiducials_SOURCE})
target_link_libraries( WarpFiducials ITKBasicFilters ITKIO ITKStatistics ${Boost_LIBRARIES} )

INSTALL(TARGETS WarpFiducials RUNTIME DESTINATION bin)

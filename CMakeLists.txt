project(FIDWARPBUILD)
cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

ExternalProject_Add(external_boost
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/External/boost-1.41.0.cmake0
  BINARY_DIR ${CMAKE_BINARY_DIR}/Build/boost
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/Install/boost
)

ExternalProject_Add(external_itk
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/External/InsightToolkit-3.20.0
  BINARY_DIR ${CMAKE_BINARY_DIR}/Build/ITK
  CMAKE_ARGS /
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_EXAMPLES=OFF /
  -DBUILD_TESTING=OFF /
  -DITK_USE_REVIEW=ON /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/Install/ITK
  LOG_BUILD 1
)

ExternalProject_Add(external_tclap
  SVN_REPOSITORY http://svn.slicer.org/Slicer3/trunk/Libs/SlicerExecutionModel/tclap
  BINARY_DIR ${CMAKE_BINARY_DIR}/Build/tclap
  CMAKE_ARGS /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/Install/tclap
  )

ExternalProject_Add(external_modparser
  DEPENDS external_tclap external_itk
  SVN_REPOSITORY http://svn.slicer.org/Slicer3/trunk/Libs/SlicerExecutionModel/ModuleDescriptionParser
  BINARY_DIR ${CMAKE_BINARY_DIR}/Build/modparser
  CMAKE_ARGS /
  -DBUILD_SHARED_LIBS=OFF
  -DITK_DIR=${CMAKE_BINARY_DIR}/Install/ITK/lib/InsightToolkit /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/Install/modparser
  )

ExternalProject_Add(external_genclp
  DEPENDS external_tclap external_modparser
  SVN_REPOSITORY http://svn.slicer.org/Slicer3/trunk/Libs/SlicerExecutionModel/GenerateCLP
  BINARY_DIR ${CMAKE_BINARY_DIR}/Build/genclp
  CMAKE_ARGS /
  -DModuleDescriptionParser_DIR=${CMAKE_BINARY_DIR}/Install/modparser/lib/ModuleDescriptionParser /
  -DTCLAP_DIR=${CMAKE_BINARY_DIR}/Install/tclap/lib/tclap /
  -DITK_DIR=${CMAKE_BINARY_DIR}/Install/ITK/lib/InsightToolkit /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/Install/genclp
  )

set ($ENV{BOOST_ROOT} "${CMAKE_BINARY_DIR}/Install/boost")

ExternalProject_Add(external_warpfid
  DEPENDS external_itk
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src
  BINARY_DIR ${CMAKE_BINARY_DIR}/Build/warpfid
  CMAKE_ARGS /
  -DCMAKE_BUILD_TYPE=Release /
  -DBOOST_ROOT=${CMAKE_BINARY_DIR}/Install/boost
  -DITK_DIR=${CMAKE_BINARY_DIR}/Install/ITK/lib/InsightToolkit /
  -DGenerateCLP_DIR=${CMAKE_BINARY_DIR}/Install/genclp/lib/GenerateCLP /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/Install/warpfid
  LOG_BUILD 1
)

#  -DBOOST_ROOT=${CMAKE_BINARY_DIR}/Install/boost
#  -DBOOST_INCLUDEDIR=${CMAKE_BINARY_DIR}/Install/boost/include/boost-1.41.0
#  -DBoost_LIBRARYDIR=${CMAKE_BINARY_DIR}/Install/boost/lib/boost-1.41.0

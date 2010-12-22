project(FIDWARPBUILD)
cmake_minimum_required(VERSION 2.8.3)

include(ExternalProject)

ExternalProject_Add(external_itk
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/External/InsightToolkit-3.20.0
  BINARY_DIR ${CMAKE_BINARY_DIR}/ITK-Build
  INSTALL_DIR ${CMAKE_BINARY_DIR}/ITK
  CMAKE_ARGS /
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_EXAMPLES=OFF /
  -DBUILD_TESTING=OFF /
  -DITK_USE_REVIEW=ON /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/ITK
  LOG_BUILD 1
)

ExternalProject_Add(external_warpfid
  DEPENDS external_itk
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/src
  BINARY_DIR ${CMAKE_BINARY_DIR}/warpfid-build
  INSTALL_DIR ${CMAKE_BINARY_DIR}/warpfid
  CMAKE_ARGS /
  -DCMAKE_BUILD_TYPE=Release
  -DITK_DIR=${CMAKE_BINARY_DIR}/ITK/lib/InsightToolkit /
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/warpfid
  LOG_BUILD 1
)



PROJECT(ITKTesting)

# The Testing directory is setup so it can build either as part of the
# normal ITK build or as a stand-alone project.  This design is useful
# for testing the installation tree of ITK.

IF(ITK_CSWIG_JAVA)
  FIND_PACKAGE(Java)
ENDIF(ITK_CSWIG_JAVA)

MAKE_DIRECTORY(${ITKTesting_BINARY_DIR}/Temporary)
IF(ITK_BINARY_DIR)
  # We are building inside the tree.
  SUBDIRS(Code)

  IF(ITK_CSWIG_TCL)
    SET(ITKWISH_EXECUTABLE "${CXX_TEST_PATH}/itkwish")
  ENDIF(ITK_CSWIG_TCL)
ELSE(ITK_BINARY_DIR)
  # We are building as a stand-alone project.

  CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
  IF(COMMAND CMAKE_POLICY)
    CMAKE_POLICY(SET CMP0003 NEW)
  ENDIF(COMMAND CMAKE_POLICY)

  SET(LIBRARY_OUTPUT_PATH ${ITKTesting_BINARY_DIR}/bin CACHE PATH "Single output directory for building all libraries.")
  SET(EXECUTABLE_OUTPUT_PATH ${ITKTesting_BINARY_DIR}/bin CACHE PATH "Single output directory for building all executables.")
  MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)
  SET(CXX_TEST_PATH ${EXECUTABLE_OUTPUT_PATH})
  GET_FILENAME_COMPONENT(ITK_SOURCE_DIR "${ITKTesting_SOURCE_DIR}" PATH)

  SET(ITK_COMPUTER_MEMORY_SIZE 1 CACHE STRING "Provide here the size of your RAM Memory in GigaBytes") 
  MARK_AS_ADVANCED(ITK_COMPUTER_MEMORY_SIZE)

  # Configure Dart testing support.
  INCLUDE(${CMAKE_ROOT}/Modules/Dart.cmake)
  MARK_AS_ADVANCED(TCL_TCLSH)
  IF(BUILD_TESTING)
    ENABLE_TESTING()
  ENDIF(BUILD_TESTING)

  # Find the data directory.
  FIND_PATH(ITK_DATA_ROOT ITKData.readme ${ITKTesting_SOURCE_DIR}/Data $ENV{ITK_DATA_ROOT})
  MARK_AS_ADVANCED(ITK_DATA_ROOT)

  # Find ITK and load its settings.
  FIND_PACKAGE(ITK)
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
    #-----------------------------------------------------------------------------
    # Test for some required system information.
    INCLUDE (${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityC.cmake)
    INCLUDE (${CMAKE_ROOT}/Modules/CMakeBackwardCompatibilityCXX.cmake)
    #-----------------------------------------------------------------------------
    SUBDIRS(Code)
    IF(ITK_CSWIG_TCL)
      FIND_PROGRAM(ITKWISH_EXECUTABLE itkwish ${ITK_DIR}/bin ${ITK_DIR})
    ENDIF(ITK_CSWIG_TCL)
    IF(ITK_CSWIG_PYTHON)
      FIND_PROGRAM(PYTHON_EXECUTABLE
        NAMES python python2.3 python2.2 python2.1 python2.0 python1.6 python1.5
        PATHS
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\2.3\\InstallPath]
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\2.2\\InstallPath]
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\2.1\\InstallPath]
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\2.0\\InstallPath]
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\1.6\\InstallPath]
        [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\1.5\\InstallPath]
        )
      MARK_AS_ADVANCED(PYTHON_EXECUTABLE)
    ENDIF(ITK_CSWIG_PYTHON)
  ELSE(ITK_FOUND)
    MESSAGE("ITK not found.  Please set ITK_DIR")
  ENDIF(ITK_FOUND)
ENDIF(ITK_BINARY_DIR)

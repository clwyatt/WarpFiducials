IF(ITK_CSWIG_TCL)
  ADD_TEST(itkBasicClasses ${ITKWISH_EXECUTABLE}
           ${ITKTesting_SOURCE_DIR}/Code/Common/itkBasicClasses.tcl)
ENDIF(ITK_CSWIG_TCL)
#-----------------------------------------------------------------------------
# Setup the system information test.  Write out some basic failsafe
# information in case the test doesn't run.

IF( NOT ITK_DISABLE_CXX_TESTING )
IF(ITK_BINARY_DIR)
  IF(TCL_TCLSH)
    ADD_TEST(PrintSelf-Common ${TCL_TCLSH}
             ${ITK_SOURCE_DIR}/Utilities/Dart/PrintSelfCheck.tcl  
             ${ITK_SOURCE_DIR}/Code/Common)
  ENDIF(TCL_TCLSH)
ENDIF(ITK_BINARY_DIR)


SET(COMMON_TESTS ${CXX_TEST_PATH}/itkCommonTests)
SET(COMMON_TESTS2 ${CXX_TEST_PATH}/itkCommonTests2)
SET(COMMON_HEADER_TEST ${CXX_TEST_PATH}/itkCommonHeaderTest)
SET(SYSTEM_INFORMATION_TEST ${CXX_TEST_PATH}/itkSystemInformationTest)
SET(SYSTEM_INFORMATION ${CXX_TEST_PATH}/itkSystemInformation)
SET(BASELINE ${ITK_DATA_ROOT}/Baseline/Common)

SET(TEMP ${ITKTesting_BINARY_DIR}/Temporary)
SET(EXAMPLES_DATA_ROOT "${ITK_SOURCE_DIR}/Examples/Data")

ADD_TEST(itkAnnulusOperatorTest ${COMMON_TESTS} itkAnnulusOperatorTest)
ADD_TEST(itkChainCodePath2DTest ${COMMON_TESTS} itkChainCodePath2DTest)
ADD_TEST(itkChainCodePathTest ${COMMON_TESTS} itkChainCodePathTest)
ADD_TEST(itkColorTableTest ${COMMON_TESTS} itkColorTableTest)
ADD_TEST(itkCommonPrintTest ${COMMON_TESTS} itkCommonPrintTest)
ADD_TEST(itkDecoratorTest ${COMMON_TESTS} itkDecoratorTest)
ADD_TEST(itkDirectoryTest ${COMMON_TESTS} itkDirectoryTest ${TEMP})
ADD_TEST(itkModifiedTimeTest ${COMMON_TESTS} itkModifiedTimeTest)
ADD_TEST(itkMultipleLogOutputTest ${COMMON_TESTS} itkMultipleLogOutputTest ${TEMP}/test_multi.txt)
ADD_TEST(itkAdaptorComparisonTest ${COMMON_TESTS} itkAdaptorComparisonTest)
ADD_TEST(itkAffineTransformTest ${COMMON_TESTS} itkAffineTransformTest)
ADD_TEST(itkFixedCenterOfRotationAffineTransformTest ${COMMON_TESTS} itkFixedCenterOfRotationAffineTransformTest)
ADD_TEST(itkFixedArrayTest ${COMMON_TESTS} itkFixedArrayTest)
ADD_TEST(itkFixedArrayTest2 ${COMMON_TESTS} itkFixedArrayTest2)
ADD_TEST(itkArrayTest ${COMMON_TESTS} itkArrayTest)
ADD_TEST(itkArray2DTest ${COMMON_TESTS} itkArray2DTest)
ADD_TEST(itkAutoPointerTest ${COMMON_TESTS} itkAutoPointerTest)
ADD_TEST(itkAzimuthElevationToCartesianTransformTest ${COMMON_TESTS} itkAzimuthElevationToCartesianTransformTest)
IF(NOT MINGW)
ADD_TEST(itkBarrierTest ${COMMON_TESTS} itkBarrierTest)
ADD_TEST(itkConditionVariableTest ${COMMON_TESTS} itkConditionVariableTest)
ADD_TEST(itkSemaphoreTest ${COMMON_TESTS2} itkSemaphoreTest)
ENDIF(NOT MINGW)
ADD_TEST(itkTimeStampTest ${COMMON_TESTS2} itkTimeStampTest)
ADD_TEST(itkBinaryThresholdImageFunctionTest ${COMMON_TESTS} itkBinaryThresholdImageFunctionTest)
ADD_TEST(itkBinaryThresholdSpatialFunctionTest ${COMMON_TESTS} itkBinaryThresholdSpatialFunctionTest)
ADD_TEST(itkBoundingBoxTest ${COMMON_TESTS} itkBoundingBoxTest)
ADD_TEST(itkBSplineDeformableTransformTest ${COMMON_TESTS} itkBSplineDeformableTransformTest)
ADD_TEST(itkBSplineInterpolationWeightFunctionTest ${COMMON_TESTS} itkBSplineInterpolationWeightFunctionTest)
ADD_TEST(itkBSplineKernelFunctionTest ${COMMON_TESTS} itkBSplineKernelFunctionTest)
ADD_TEST(itkBoundaryConditionTest ${COMMON_TESTS} itkBoundaryConditionTest)
ADD_TEST(itkByteSwapTest ${COMMON_TESTS} itkByteSwapTest)
ADD_TEST(itkCenteredRigid2DTransformTest ${COMMON_TESTS} itkCenteredRigid2DTransformTest ${TEMP}/testCenteredTransform.txt)
ADD_TEST(itkCenteredAffineTransformTest ${COMMON_TESTS} itkCenteredAffineTransformTest)
ADD_TEST(itkCenteredTransformInitializerTest ${COMMON_TESTS} itkCenteredTransformInitializerTest)
ADD_TEST(itkCenteredVersorTransformInitializerTest ${COMMON_TESTS} itkCenteredVersorTransformInitializerTest)
ADD_TEST(itkCentralDifferenceImageFunctionTest ${COMMON_TESTS} itkCentralDifferenceImageFunctionTest)
ADD_TEST(itkCenteredEuler3DTransformTest ${COMMON_TESTS} itkCenteredEuler3DTransformTest)
ADD_TEST(itkConstNeighborhoodIteratorTest ${COMMON_TESTS} itkConstNeighborhoodIteratorTest)
ADD_TEST(itkConstShapedNeighborhoodIteratorTest ${COMMON_TESTS} itkConstShapedNeighborhoodIteratorTest)
ADD_TEST(itkCovariantVectorGeometryTest ${COMMON_TESTS} itkCovariantVectorGeometryTest)
ADD_TEST(itkCovarianceImageFunctionTest ${COMMON_TESTS} itkCovarianceImageFunctionTest)
ADD_TEST(itkDataTypeTest ${COMMON_TESTS} itkDataTypeTest)
ADD_TEST(itkDynamicMeshTest ${COMMON_TESTS} itkDynamicMeshTest)
ADD_TEST(itkDiffusionTensor3DTest ${COMMON_TESTS} itkDiffusionTensor3DTest)
ADD_TEST(itkEuler2DTransformTest ${COMMON_TESTS} itkEuler2DTransformTest ${TEMP}/testEulerTransform.txt)
ADD_TEST(itkEuler3DTransformTest ${COMMON_TESTS} itkEuler3DTransformTest)
ADD_TEST(itkEventObjectTest ${COMMON_TESTS} itkEventObjectTest)
ADD_TEST(itkExceptionObjectTest ${COMMON_TESTS} itkExceptionObjectTest)
ADD_TEST(itkFileOutputWindowTest ${COMMON_TESTS} itkFileOutputWindowTest)
ADD_TEST(itkFiniteCylinderSpatialFunctionTest ${COMMON_TESTS} itkFiniteCylinderSpatialFunctionTest)
ADD_TEST(itkFloodFilledSpatialFunctionTest ${COMMON_TESTS} itkFloodFilledSpatialFunctionTest)
ADD_TEST(itkFourierSeriesPathTest ${COMMON_TESTS} itkFourierSeriesPathTest)
ADD_TEST(itkGaussianBlurImageFunctionTest ${COMMON_TESTS} itkGaussianBlurImageFunctionTest)
ADD_TEST(itkGaussianDerivativeImageFunctionTest ${COMMON_TESTS} itkGaussianDerivativeImageFunctionTest)
ADD_TEST(itkGaussianSpatialFunctionTest ${COMMON_TESTS} itkGaussianSpatialFunctionTest)
ADD_TEST(itkCommonHeaderTest ${COMMON_HEADER_TEST})
ADD_TEST(itkHashTableTest ${COMMON_TESTS} itkHashTableTest)
ADD_TEST(itkIdentityTransformTest ${COMMON_TESTS} itkIdentityTransformTest)
ADD_TEST(itkImageAdaptorTest ${COMMON_TESTS} itkImageAdaptorTest)
ADD_TEST(itkOrientedImageAdaptorTest ${COMMON_TESTS} itkOrientedImageAdaptorTest)
ADD_TEST(itkImageAdaptorInterpolateImageFunctionTest ${COMMON_TESTS} itkImageAdaptorInterpolateImageFunctionTest)

IF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest2 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest2PixelCentered.png
            ${TEMP}/itkBSplineDeformableTransformTest2PixelCentered.png
  itkBSplineDeformableTransformTest2
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements5.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest2PixelCentered.png
        ${TEMP}/itkBSplineDeformableTransformTest2DeformationFieldPixelCentered.mhd 1
  )
ELSE(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest2 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest2.png
            ${TEMP}/itkBSplineDeformableTransformTest2.png
  itkBSplineDeformableTransformTest2
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements4.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest2.png
        ${TEMP}/itkBSplineDeformableTransformTest2DeformationField.mhd 1
  )
ENDIF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)

IF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest3 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest3PixelCentered.png
            ${TEMP}/itkBSplineDeformableTransformTest3PixelCentered.png
  itkBSplineDeformableTransformTest2
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements3.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest3PixelCentered.png
        ${TEMP}/itkBSplineDeformableTransformTest3DeformationFieldPixelCentered.mhd 2
  )
ELSE(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest3 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest3.png
            ${TEMP}/itkBSplineDeformableTransformTest3.png
  itkBSplineDeformableTransformTest2
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements3.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest3.png
        ${TEMP}/itkBSplineDeformableTransformTest3DeformationField.mhd 2
  )
ENDIF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)

IF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest4 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest4PixelCentered.png
            ${TEMP}/itkBSplineDeformableTransformTest4PixelCentered.png
  itkBSplineDeformableTransformTest2
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest4PixelCentered.png
        ${TEMP}/itkBSplineDeformableTransformTest4DeformationFieldPixelCentered.mhd 3
  )
ELSE(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest4 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest4.png
            ${TEMP}/itkBSplineDeformableTransformTest4.png
  itkBSplineDeformableTransformTest2
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest4.png
        ${TEMP}/itkBSplineDeformableTransformTest4DeformationField.mhd 3
  )
ENDIF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)

IF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest5 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest5PixelCentered.png
            ${TEMP}/itkBSplineDeformableTransformTest5PixelCentered.png
  itkBSplineDeformableTransformTest3
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest5PixelCentered.png
        ${TEMP}/itkBSplineDeformableTransformTest5DeformationFieldPixelCentered.mhd 0
  )
ELSE(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest5 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest5.png
            ${TEMP}/itkBSplineDeformableTransformTest5.png
  itkBSplineDeformableTransformTest3
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest5.png
        ${TEMP}/itkBSplineDeformableTransformTest5DeformationField.mhd 0
  )
ENDIF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)

IF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest6 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest5PixelCentered.png
            ${TEMP}/itkBSplineDeformableTransformTest6PixelCentered.png
  itkBSplineDeformableTransformTest3
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest6PixelCentered.png
        ${TEMP}/itkBSplineDeformableTransformTest6DeformationFieldPixelCentered.mhd 1
  )
ELSE(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest6 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest5.png
            ${TEMP}/itkBSplineDeformableTransformTest6.png
  itkBSplineDeformableTransformTest3
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest6.png
        ${TEMP}/itkBSplineDeformableTransformTest6DeformationField.mhd 1
  )
ENDIF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)

IF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest7 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest5PixelCentered.png
            ${TEMP}/itkBSplineDeformableTransformTest7PixelCentered.png
  itkBSplineDeformableTransformTest3
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest7PixelCentered.png
        ${TEMP}/itkBSplineDeformableTransformTest7DeformationFieldPixelCentered.mhd 2
  )
ELSE(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)
ADD_TEST(itkBSplineDeformableTransformTest7 ${COMMON_TESTS2} 
  --compare ${BASELINE}/itkBSplineDeformableTransformTest5.png
            ${TEMP}/itkBSplineDeformableTransformTest7.png
  itkBSplineDeformableTransformTest3
        ${ITK_SOURCE_DIR}/Examples/Data/BSplineDisplacements1.txt
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${ITK_SOURCE_DIR}/Examples/Data/DiagonalLines.png
        ${TEMP}/itkBSplineDeformableTransformTest7.png
        ${TEMP}/itkBSplineDeformableTransformTest7DeformationField.mhd 2
  )
ENDIF(ITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY)

ADD_TEST(itkCellInterfaceTest ${COMMON_TESTS2} itkCellInterfaceTest)
ADD_TEST(itkImageTransformTest ${COMMON_TESTS2} itkImageTransformTest)
ADD_TEST(itkImageToImageFilterTest ${COMMON_TESTS2} itkImageToImageFilterTest)
ADD_TEST(itkRGBInterpolateImageFunctionTest ${COMMON_TESTS2} itkRGBInterpolateImageFunctionTest)
ADD_TEST(itkImageDuplicatorTest ${COMMON_TESTS} itkImageDuplicatorTest)
ADD_TEST(itkImageIteratorTest ${COMMON_TESTS} itkImageIteratorTest)
ADD_TEST(itkImageIteratorsForwardBackwardTest ${COMMON_TESTS} itkImageIteratorsForwardBackwardTest)
ADD_TEST(itkImageIteratorWithIndexTest ${COMMON_TESTS} itkImageIteratorWithIndexTest)
ADD_TEST(itkImageRegionIteratorTest ${COMMON_TESTS} itkImageRegionIteratorTest)
ADD_TEST(itkImageLinearIteratorTest ${COMMON_TESTS} itkImageLinearIteratorTest)
ADD_TEST(itkImageRandomIteratorTest ${COMMON_TESTS} itkImageRandomIteratorTest)
ADD_TEST(itkImageRandomIteratorTest2 ${COMMON_TESTS} itkImageRandomIteratorTest2
      ${TEMP}/itkImageRandomIteratorTest2Output.mha)
ADD_TEST(itkImageRandomNonRepeatingIteratorWithIndexTest ${COMMON_TESTS} itkImageRandomNonRepeatingIteratorWithIndexTest)
ADD_TEST(itkImageRegionTest ${COMMON_TESTS} itkImageRegionTest)
ADD_TEST(itkImageRegionExclusionIteratorWithIndexTest ${COMMON_TESTS} itkImageRegionExclusionIteratorWithIndexTest)
ADD_TEST(itkImageReverseIteratorTest ${COMMON_TESTS} itkImageReverseIteratorTest)
ADD_TEST(itkImageSliceIteratorTest ${COMMON_TESTS} itkImageSliceIteratorTest)
ADD_TEST(itkImageTest ${COMMON_TESTS} itkImageTest)
ADD_TEST(itkIteratorTests ${COMMON_TESTS} itkIteratorTests)
ADD_TEST(itkLandmarkBasedTransformInitializerTest ${COMMON_TESTS} itkLandmarkBasedTransformInitializerTest)
ADD_TEST(itkLightObjectTest ${COMMON_TESTS} itkLightObjectTest)
ADD_TEST(itkLinearInterpolateImageFunctionTest ${COMMON_TESTS2} itkLinearInterpolateImageFunctionTest)
ADD_TEST(itkLineIteratorTest ${COMMON_TESTS} 
  --compare ${BASELINE}/itkLineIteratorTest.png
            ${TEMP}/itkLineIteratorTest.png
  itkLineIteratorTest
            ${TEMP}/itkLineIteratorTest.png
)
ADD_TEST(itkLevelSetFunctionTest ${COMMON_TESTS} itkLevelSetFunctionTest)
ADD_TEST(itkLoggerTest ${COMMON_TESTS} itkLoggerTest ${TEMP}/test_logger.txt)
#Needs to be added to SRCS when this works across platforms.  itkLoggerThreadWrapperTest.cxx
#ADD_TEST(itkLoggerThreadWrapperTest ${COMMON_TESTS} itkLoggerThreadWrapperTest ${TEMP}/test_logger.txt)
ADD_TEST(itkLoggerOutputTest ${COMMON_TESTS} itkLoggerOutputTest ${TEMP}/test_loggerOutput.txt)
ADD_TEST(itkLoggerManagerTest ${COMMON_TESTS} itkLoggerManagerTest ${TEMP}/test_LoggerManager.txt)
ADD_TEST(itkMatrixTest ${COMMON_TESTS} itkMatrixTest)
ADD_TEST(itkMapContainerTest ${COMMON_TESTS} itkMapContainerTest)
ADD_TEST(itkMahalanobisDistanceThresholdImageFunctionTest ${COMMON_TESTS} itkMahalanobisDistanceThresholdImageFunctionTest)

ADD_TEST(itkMaximumDecisionRuleTest ${COMMON_TESTS2} itkMaximumDecisionRuleTest)
ADD_TEST(itkMaximumRatioDecisionRuleTest ${COMMON_TESTS2} itkMaximumRatioDecisionRuleTest)
ADD_TEST(itkMeanImageFunctionTest ${COMMON_TESTS2} itkMeanImageFunctionTest)

IF(ITK_DATA_ROOT)
  ADD_TEST(itkDifferenceImageFilterTest ${COMMON_TESTS2} 
       --compare ${BASELINE}/itkDifferenceImageFilterTest.png 
                 ${ITK_TEST_OUTPUT_DIR}/itkDifferenceImageFilterTest.png 
     itkDifferenceImageFilterTest 
                 ${ITK_DATA_ROOT}/Input/cake_easy.png 
                 ${ITK_DATA_ROOT}/Input/cake_hard.png 
                 ${ITK_TEST_OUTPUT_DIR}/itkDifferenceImageFilterTest.png 10 1)

  IF(ITK_USE_ORIENTED_IMAGE_DIRECTION)
    # The difference between these two tests is that if the CMake flag for
    # image orientation is ON, then the second computation of the gradient
    # should now take direction into account. Since not all builds have this
    # flag enabled, the builds with the option OFF will get the gradients in 
    # the frame of the image grid.
    ADD_TEST(itkOrientedImage2DTest01 ${COMMON_TESTS2} 
       itkOrientedImage2DTest 
                   ${EXAMPLES_DATA_ROOT}/BrainProtonDensitySliceBorder20DirectionPlus30.mhd

                       0.000    0.000
                     191.392  110.500
                    -128.500  222.569

                       4.000   -8.000
                       7.4641  -4.9282
                    )
  ELSE(ITK_USE_ORIENTED_IMAGE_DIRECTION)
    ADD_TEST(itkOrientedImage2DTest01 ${COMMON_TESTS2} 
       itkOrientedImage2DTest 
                   ${EXAMPLES_DATA_ROOT}/BrainProtonDensitySliceBorder20DirectionPlus30.mhd

                       0.000    0.000
                     191.392  110.500
                    -128.500  222.569

                       4.000   -8.000
                       4.000   -8.000
                    )
  ENDIF(ITK_USE_ORIENTED_IMAGE_DIRECTION)


  IF(ITK_USE_ORIENTED_IMAGE_DIRECTION)
    # The difference between these two tests is that if the CMake flag for
    # image orientation is ON, then the second computation of the gradient
    # should now take direction into account. Since not all builds have this
    # flag enabled, the builds with the option OFF will get the gradients in 
    # the frame of the image grid.
    ADD_TEST(itkOrientedImage3DTest01 ${COMMON_TESTS2} 
       itkOrientedImage3DTest 
                   ${ITK_DATA_ROOT}/Input/HeadMRVolumeWithDirection.mhd
                      0.000      0.000     0.000
                    166.277    -96.000     0.000
                    124.000    214.774     0.000
                      0.000      0.000   168.000

                      0.125     -4.125     9.000
                     -1.95425   -3.63485   9.000
                   )
  ELSE(ITK_USE_ORIENTED_IMAGE_DIRECTION)
    ADD_TEST(itkOrientedImage3DTest01 ${COMMON_TESTS2} 
       itkOrientedImage3DTest 
                   ${ITK_DATA_ROOT}/Input/HeadMRVolumeWithDirection.mhd
                      0.000      0.000     0.000
                    166.277    -96.000     0.000
                    124.000    214.774     0.000
                      0.000      0.000   168.000

                      0.125     -4.125     9.000
                      0.125     -4.125     9.000
                   )
  ENDIF(ITK_USE_ORIENTED_IMAGE_DIRECTION)


ENDIF(ITK_DATA_ROOT)

ADD_TEST(itkOrientedImageProfileTest1 ${COMMON_TESTS2} itkOrientedImageProfileTest1)
ADD_TEST(itkOrientedImageProfileTest2 ${COMMON_TESTS2} itkOrientedImageProfileTest2)
ADD_TEST(itkOrientedImageProfileTest3 ${COMMON_TESTS2} itkOrientedImageProfileTest3)

ADD_TEST(itkMathRoundTest ${COMMON_TESTS2} itkMathRoundTest)
ADD_TEST(itkMathRoundTest2 ${COMMON_TESTS2} itkMathRoundTest2)

ADD_TEST(itkVNLRoundProfileTest1 ${COMMON_TESTS2} itkVNLRoundProfileTest1)
ADD_TEST(itkMathRoundProfileTest1 ${COMMON_TESTS2} itkMathRoundProfileTest1)

ADD_TEST(itkMathCastWithRangeCheckTest ${COMMON_TESTS2} itkMathCastWithRangeCheckTest )

ADD_TEST(itkMedianImageFunctionTest ${COMMON_TESTS2} itkMedianImageFunctionTest)
ADD_TEST(itkMedialNodeCorrespondencesTest ${COMMON_TESTS2} itkMedialNodeCorrespondencesTest)

#
# This test should be enabled if you suspect that the memory leak detector
# is not working
IF(0)
  ADD_TEST(itkMemoryLeakTest ${COMMON_TESTS2} itkMemoryLeakTest)
ENDIF(0)

ADD_TEST(itkMeshTest ${COMMON_TESTS2} itkMeshTest)
ADD_TEST(itkMeshFstreamTest ${COMMON_TESTS2} itkMeshFstreamTest ${TEMP}/testMeshFstream.txt)
ADD_TEST(itkMeshSourceGraftOutputTest ${COMMON_TESTS2} itkMeshSourceGraftOutputTest)
ADD_TEST(itkMetaDataDictionaryTest ${COMMON_TESTS2} itkMetaDataDictionaryTest)
ADD_TEST(itkMinimumDecisionRuleTest ${COMMON_TESTS2} itkMinimumDecisionRuleTest)
ADD_TEST(itkMultiThreaderTest ${COMMON_TESTS2} itkMultiThreaderTest)
ADD_TEST(itkNearestNeighborExtrapolateImageFunctionTest ${COMMON_TESTS2} itkNearestNeighborExtrapolateImageFunctionTest)
ADD_TEST(itkNeighborhoodTest ${COMMON_TESTS2} itkNeighborhoodTest)
ADD_TEST(itkNeighborhoodIteratorTest ${COMMON_TESTS2} itkNeighborhoodIteratorTest)
ADD_TEST(itkNeighborhoodOperatorImageFunctionTest ${COMMON_TESTS2} itkNeighborhoodOperatorImageFunctionTest)
ADD_TEST(itkNeighborhoodOperatorTest ${COMMON_TESTS2} itkNeighborhoodOperatorTest)
ADD_TEST(itkNumericTraitsTest ${COMMON_TESTS2} itkNumericTraitsTest)
ADD_TEST(itkIntTypesTest ${COMMON_TESTS2} itkIntTypesTest)
ADD_TEST(itkOctreeTest ${COMMON_TESTS2} itkOctreeTest)
ADD_TEST(itkObjectFactoryTest ${COMMON_TESTS2} itkObjectFactoryTest)
IF(ITK_BUILD_SHARED_LIBS)
ADD_TEST(itkObjectFactoryTest2 ${COMMON_TESTS2} itkObjectFactoryTest2 ${ITK_TEST_OUTPUT_DIR})
SET_TESTS_PROPERTIES(itkObjectFactoryTest2 PROPERTIES RUN_SERIAL 1)

ENDIF(ITK_BUILD_SHARED_LIBS)
ADD_TEST(itkOrthogonallyCorrected2DParametricPathTest ${COMMON_TESTS2} itkOrthogonallyCorrected2DParametricPathTest)
ADD_TEST(itkPathFunctionsTest ${COMMON_TESTS2} itkPathFunctionsTest)
ADD_TEST(itkNonUniformBSplineTest ${COMMON_TESTS2} itkNonUniformBSplineTest)
ADD_TEST(itkPathIteratorTest ${COMMON_TESTS2} itkPathIteratorTest)
ADD_TEST(itkPCAShapeSignedDistanceFunctionTest ${COMMON_TESTS2} itkPCAShapeSignedDistanceFunctionTest)
ADD_TEST(itkPeriodicBoundaryConditionTest ${COMMON_TESTS2} itkPeriodicBoundaryConditionTest)
ADD_TEST(itkPhasedArray3DSpecialCoordinatesImageTest ${COMMON_TESTS2} itkPhasedArray3DSpecialCoordinatesImageTest)
ADD_TEST(itkPixelAccessTest ${COMMON_TESTS2} itkPixelAccessTest)
ADD_TEST(itkPointGeometryTest ${COMMON_TESTS2} itkPointGeometryTest)
ADD_TEST(itkPointSetTest ${COMMON_TESTS2} itkPointSetTest)
ADD_TEST(itkPolyLineParametricPathTest ${COMMON_TESTS2} itkPolyLineParametricPathTest)
ADD_TEST(itkPolygonCellTest ${COMMON_TESTS2} itkPolygonCellTest)
ADD_TEST(itkQuaternionRigidTransformTest ${COMMON_TESTS2} itkQuaternionRigidTransformTest)
ADD_TEST(itkRealTimeClockTest ${COMMON_TESTS2} itkRealTimeClockTest)
ADD_TEST(itkRGBPixelTest ${COMMON_TESTS2} itkRGBPixelTest)
ADD_TEST(itkRGBInterpolateImageFunctionTest ${COMMON_TESTS2} itkRGBInterpolateImageFunctionTest)
ADD_TEST(itkRGBToVectorImageAdaptorTest ${COMMON_TESTS2} itkRGBToVectorImageAdaptorTest)
ADD_TEST(itkRigid2DTransformTest ${COMMON_TESTS2} itkRigid2DTransformTest)
ADD_TEST(itkRigid3DTransformTest ${COMMON_TESTS2} itkRigid3DTransformTest)
ADD_TEST(itkRigid3DPerspectiveTransformTest ${COMMON_TESTS2} itkRigid3DPerspectiveTransformTest)
ADD_TEST(itkScalarToRGBPixelFunctorTest ${COMMON_TESTS2} itkScalarToRGBPixelFunctorTest)
ADD_TEST(itkScatterMatrixImageFunctionTest ${COMMON_TESTS2} itkScatterMatrixImageFunctionTest)
ADD_TEST(itkScaleSkewVersor3DTransformTest ${COMMON_TESTS2} itkScaleSkewVersor3DTransformTest)
ADD_TEST(itkScaleVersor3DTransformTest ${COMMON_TESTS2} itkScaleVersor3DTransformTest)
ADD_TEST(itkScaleTransformTest ${COMMON_TESTS2} itkScaleTransformTest)
ADD_TEST(itkScaleLogarithmicTransformTest ${COMMON_TESTS2} itkScaleLogarithmicTransformTest)
ADD_TEST(itkShapedNeighborhoodIteratorTest ${COMMON_TESTS2} itkShapedNeighborhoodIteratorTest)
ADD_TEST(itkSimilarity2DTransformTest ${COMMON_TESTS2} itkSimilarity2DTransformTest)
ADD_TEST(itkSimilarity3DTransformTest ${COMMON_TESTS2} itkSimilarity3DTransformTest)
ADD_TEST(itkSimpleFilterWatcherTest ${COMMON_TESTS2} itkSimpleFilterWatcherTest)
ADD_TEST(itkSimplexMeshTest ${COMMON_TESTS2} itkSimplexMeshTest)
ADD_TEST(itkSizeTest ${COMMON_TESTS2} itkSizeTest)
ADD_TEST(itkSliceIteratorTest ${COMMON_TESTS2} itkSliceIteratorTest)
ADD_TEST(itkSmartPointerTest ${COMMON_TESTS2} itkSmartPointerTest)
ADD_TEST(itkSparseImageTest ${COMMON_TESTS2} itkSparseImageTest)
ADD_TEST(itkSpatialFunctionTest ${COMMON_TESTS2} itkSpatialFunctionTest)
ADD_TEST(itkSphereSignedDistanceFunctionTest ${COMMON_TESTS2} itkSphereSignedDistanceFunctionTest)
ADD_TEST(itkSplineKernelTransformTest ${COMMON_TESTS2} itkSplineKernelTransformTest)
ADD_TEST(itkSTLContainerAdaptorTest ${COMMON_TESTS2} itkSTLContainerAdaptorTest)
ADD_TEST(itkStdStreamLogOutputTest ${COMMON_TESTS2} itkStdStreamLogOutputTest ${TEMP}/testStreamLogOutput.txt)
ADD_TEST(itkThreadDefsTest ${COMMON_TESTS2} itkThreadDefsTest)
ADD_TEST(itkThreadLoggerTest ${COMMON_TESTS2} itkThreadLoggerTest ${TEMP}/test_threadLogger.txt)
ADD_TEST(itkTimeProbesTest ${COMMON_TESTS2} itkTimeProbesTest)
ADD_TEST(itkTransformTest ${COMMON_TESTS2} itkTransformTest)
ADD_TEST(itkTransformFactoryBaseTest ${COMMON_TESTS2} itkTransformFactoryBaseTest)
ADD_TEST(itkTransformsSetParametersTest ${COMMON_TESTS2} itkTransformsSetParametersTest)
ADD_TEST(itkTranslationTransformTest ${COMMON_TESTS2} itkTranslationTransformTest)
ADD_TEST(itkTreeContainerTest ${COMMON_TESTS2} itkTreeContainerTest)
ADD_TEST(itkTreeContainerTest2 ${COMMON_TESTS2} itkTreeContainerTest2)
ADD_TEST(itkTriangleCellTest ${COMMON_TESTS2} itkTriangleCellTest)
ADD_TEST(itkVarianceImageFunctionTest ${COMMON_TESTS2} itkVarianceImageFunctionTest)
ADD_TEST(itkVectorGeometryTest ${COMMON_TESTS2} itkVectorGeometryTest)
ADD_TEST(itkVectorImageTest ${COMMON_TESTS2} itkVectorImageTest ${TEMP}/VectorImage.nrrd ${TEMP}/VectorImage.mhd)
ADD_TEST(itkVectorMeanImageFunctionTest ${COMMON_TESTS2} itkVectorMeanImageFunctionTest)
ADD_TEST(itkVersorTransformTest ${COMMON_TESTS2} itkVersorTransformTest)
ADD_TEST(itkVersorTest ${COMMON_TESTS2} itkVersorTest)
ADD_TEST(itkVersorRigid3DTransformTest ${COMMON_TESTS2} itkVersorRigid3DTransformTest)
ADD_TEST(itkVectorTest ${COMMON_TESTS2} itkVectorTest)
ADD_TEST(itkVectorInterpolateImageFunctionTest ${COMMON_TESTS2} itkVectorInterpolateImageFunctionTest)
ADD_TEST(itkVectorToRGBImageAdaptorTest ${COMMON_TESTS2} itkVectorToRGBImageAdaptorTest)
ADD_TEST(itkWindowedSincInterpolateImageFunctionTest ${COMMON_TESTS2} itkWindowedSincInterpolateImageFunctionTest)
ADD_TEST(itkEllipsoidInteriorExteriorSpatialFunctionTest ${COMMON_TESTS2} itkEllipsoidInteriorExteriorSpatialFunctionTest)
ADD_TEST(itkSymmetricEllipsoidInteriorExteriorSpatialFunctionTest ${COMMON_TESTS2} itkSymmetricEllipsoidInteriorExteriorSpatialFunctionTest)
ADD_TEST(itkSymmetricSecondRankTensorTest ${COMMON_TESTS2} itkSymmetricSecondRankTensorTest)
ADD_TEST(itkSymmetricEigenAnalysisTest ${COMMON_TESTS2} itkSymmetricEigenAnalysisTest)
ADD_TEST(itkSTLThreadTest ${COMMON_TESTS2} itkSTLThreadTest)
ADD_TEST(itkVariableLengthVectorTest ${COMMON_TESTS2} itkVariableLengthVectorTest)
ADD_TEST(itkVariableSizeMatrixTest ${COMMON_TESTS2} itkVariableSizeMatrixTest)
ADD_TEST(itkImportContainerTest ${COMMON_TESTS2} itkImportContainerTest)
#ADD_TEST(itkQuaternionOrientationAdapterTest ${COMMON_TESTS2} itkQuaternionOrientationAdapterTest)
ADD_TEST(itkZeroFluxBoundaryConditionTest ${COMMON_TESTS2} itkZeroFluxBoundaryConditionTest)
ADD_TEST(itkSystemInformationTest ${SYSTEM_INFORMATION_TEST})
ADD_TEST(SystemInformation ${SYSTEM_INFORMATION})
IF( ${ITK_COMPUTER_MEMORY_SIZE} GREATER 4 )
  ADD_TEST(itkImageFillBufferTest4.1 ${COMMON_TESTS2} itkImageFillBufferTest 4.1)
ENDIF( ${ITK_COMPUTER_MEMORY_SIZE} GREATER 4 )


IF(ITK_BINARY_DIR)
  SET(ITK_SYSTEM_INFORMATION_DIR ${ITK_BINARY_DIR})
ELSE(ITK_BINARY_DIR)
  SET(ITK_SYSTEM_INFORMATION_DIR ${ITKTesting_BINARY_DIR})
ENDIF(ITK_BINARY_DIR)

WRITE_FILE(
  "${ITK_SYSTEM_INFORMATION_DIR}/Testing/HTML/TestingResults/Sites/${SITE}/${BUILDNAME}/BuildNameNotes.xml"

  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
  "<Site BuildName=\"CMakeCache\" Name=\"crd\">\n"
  "<BuildNameNotes>\n"
  "<Note>\n"
  "<DateTime>Wed Oct 24 1:00:00 EST</DateTime>\n"
  "<Text>\n"
  "The full itkSystemInformationTest did not run.\n"
  "Here is some basic information:\n"
  "ITK_SOURCE_DIR = ${ITK_SOURCE_DIR}\n"
  "ITK_BINARY_DIR = ${ITK_BINARY_DIR}\n"
  "ITKTesting_SOURCE_DIR = ${ITKTesting_SOURCE_DIR}\n"
  "ITKTesting_BINARY_DIR = ${ITKTesting_BINARY_DIR}\n"
  "CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}\n"
  "CMAKE_C_FLAGS = ${CMAKE_C_FLAGS}\n"
  "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}\n"
  "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}\n"
  "CMAKE_SYSTEM = ${CMAKE_SYSTEM}\n"
  "CMAKE_MAKE_PROGRAM = ${CMAKE_MAKE_PROGRAM}\n"
  "</Text>\n"
  "</Note>\n"
  "</BuildNameNotes>\n"
  "</Site>\n"
)
#-----------------------------------------------------------------------------
SET(BasicTests_SRCS
itkImageComputeOffsetAndIndexTest.cxx
itkCommonPrintTest.cxx
itkConditionVariableTest.cxx
itkModifiedTimeTest.cxx
itkMultipleLogOutputTest.cxx
itkAdaptorComparisonTest.cxx
itkAffineTransformTest.cxx
itkAnnulusOperatorTest.cxx
itkArrayTest.cxx
itkArray2DTest.cxx
itkAutoPointerTest.cxx
itkAzimuthElevationToCartesianTransformTest.cxx
itkBarrierTest.cxx
itkBinaryThresholdImageFunctionTest.cxx
itkBinaryThresholdSpatialFunctionTest.cxx
itkBoundingBoxTest.cxx
itkBSplineDeformableTransformTest.cxx
itkBSplineInterpolationWeightFunctionTest.cxx
itkBSplineKernelFunctionTest.cxx
itkBoundaryConditionTest.cxx
itkByteSwapTest.cxx
itkCenteredRigid2DTransformTest.cxx
itkCenteredAffineTransformTest.cxx
itkCentralDifferenceImageFunctionTest.cxx
itkCenteredEuler3DTransformTest.cxx
itkCenteredTransformInitializerTest.cxx
itkCenteredVersorTransformInitializerTest.cxx
itkChainCodePath2DTest.cxx
itkChainCodePathTest.cxx
itkColorTableTest.cxx
itkConstNeighborhoodIteratorTest.cxx
itkConstShapedNeighborhoodIteratorTest.cxx
itkCovariantVectorGeometryTest.cxx
itkCovarianceImageFunctionTest.cxx
itkDataTypeTest.cxx
itkDecoratorTest.cxx
itkDirectoryTest.cxx
itkDynamicMeshTest.cxx
itkDiffusionTensor3DTest.cxx
itkEuler2DTransformTest.cxx
itkEuler3DTransformTest.cxx
itkEventObjectTest.cxx
itkExceptionObjectTest.cxx
itkFileOutputWindowTest.cxx
itkFiniteCylinderSpatialFunctionTest.cxx
itkFixedArrayTest.cxx
itkFixedArrayTest2.cxx
itkFixedCenterOfRotationAffineTransformTest.cxx
itkFloodFilledSpatialFunctionTest.cxx
itkFourierSeriesPathTest.cxx
itkGaussianBlurImageFunctionTest.cxx
itkGaussianDerivativeImageFunctionTest.cxx
itkGaussianSpatialFunctionTest.cxx
itkHashTableTest.cxx
itkIdentityTransformTest.cxx
itkImageAdaptorTest.cxx
itkOrientedImageAdaptorTest.cxx
itkImageAdaptorInterpolateImageFunctionTest.cxx
itkImageDuplicatorTest.cxx
itkImageIteratorTest.cxx
itkImageIteratorsForwardBackwardTest.cxx
itkImageIteratorWithIndexTest.cxx
itkImageRegionIteratorTest.cxx
itkImageLinearIteratorTest.cxx
itkImageRandomIteratorTest.cxx
itkImageRandomIteratorTest2.cxx
itkImageRandomNonRepeatingIteratorWithIndexTest.cxx
itkImageRegionTest.cxx
itkImageRegionExclusionIteratorWithIndexTest.cxx
itkImageReverseIteratorTest.cxx
itkImageSliceIteratorTest.cxx
itkImageTest.cxx
itkIteratorTests.cxx
itkLandmarkBasedTransformInitializerTest.cxx
itkLevelSetFunctionTest.cxx
itkLightObjectTest.cxx
itkLineIteratorTest.cxx
itkLoggerTest.cxx
itkLoggerOutputTest.cxx
itkLoggerManagerTest.cxx
itkMatrixTest.cxx
itkMapContainerTest.cxx
itkMahalanobisDistanceThresholdImageFunctionTest.cxx
)


#-----------------------------------------------------------------------------
SET(BasicTests2_SRCS
itkBSplineDeformableTransformTest2.cxx 
itkBSplineDeformableTransformTest3.cxx 
itkCellInterfaceTest.cxx
itkImageTransformTest.cxx
itkImageToImageFilterTest.cxx
itkLinearInterpolateImageFunctionTest.cxx
itkMaximumDecisionRuleTest.cxx
itkMaximumRatioDecisionRuleTest.cxx
itkDifferenceImageFilterTest.cxx 
itkMeanImageFunctionTest.cxx 
itkMedialNodeCorrespondencesTest.cxx
itkMedianImageFunctionTest.cxx 
itkMemoryLeakTest.cxx
itkMeshTest.cxx
itkMeshFstreamTest.cxx
itkMeshSourceGraftOutputTest.cxx
itkMetaDataDictionaryTest.cxx
itkMinimumDecisionRuleTest.cxx
itkMultiThreaderTest.cxx
itkNearestNeighborExtrapolateImageFunctionTest.cxx
itkNeighborhoodTest.cxx
itkNeighborhoodIteratorTest.cxx
itkNeighborhoodOperatorImageFunctionTest.cxx
itkNeighborhoodOperatorTest.cxx
itkNumericTraitsTest.cxx
itkIntTypesTest.cxx
itkObjectFactoryTest.cxx
itkObjectFactoryTest2.cxx
itkObjectStoreTest.cxx
itkOctreeTest.cxx
itkOrthogonallyCorrected2DParametricPathTest.cxx
itkOrientedImage2DTest.cxx
itkOrientedImage3DTest.cxx
itkOrientedImageProfileTest1.cxx
itkOrientedImageProfileTest2.cxx
itkOrientedImageProfileTest3.cxx
itkVNLRoundProfileTest1.cxx
itkMathRoundTest.cxx
itkMathRoundTest2.cxx
itkMathRoundProfileTest1.cxx
itkMathCastWithRangeCheck.cxx
itkNonUniformBSplineTest.cxx
itkPathFunctionsTest.cxx
itkPathIteratorTest.cxx
itkPCAShapeSignedDistanceFunctionTest.cxx
itkPeriodicBoundaryConditionTest.cxx
itkPhasedArray3DSpecialCoordinatesImageTest.cxx
itkPixelAccessTest.cxx
itkPointGeometryTest.cxx
itkPointSetTest.cxx
itkPolyLineParametricPathTest.cxx
itkPolygonCellTest.cxx
itkQuaternionRigidTransformTest.cxx
itkRealTimeClockTest.cxx
itkRGBPixelTest.cxx
itkRGBInterpolateImageFunctionTest.cxx
itkRGBToVectorImageAdaptorTest.cxx
itkRigid2DTransformTest.cxx
itkRigid3DTransformTest.cxx
itkRigid3DPerspectiveTransformTest.cxx
itkScalarToRGBPixelFunctorTest.cxx
itkScatterMatrixImageFunctionTest.cxx
itkScaleSkewVersor3DTransformTest.cxx
itkScaleVersor3DTransformTest.cxx
itkScaleTransformTest.cxx
itkScaleLogarithmicTransformTest.cxx
itkSemaphoreTest.cxx
itkShapedNeighborhoodIteratorTest.cxx
itkSimilarity2DTransformTest.cxx
itkSimilarity3DTransformTest.cxx
itkSimpleFilterWatcherTest.cxx
itkSimplexMeshTest.cxx
itkSizeTest.cxx
itkSliceIteratorTest.cxx
itkSmartPointerTest.cxx
itkSparseImageTest.cxx
itkSpatialFunctionTest.cxx
itkSphereSignedDistanceFunctionTest.cxx
itkSplineKernelTransformTest.cxx
itkSTLContainerAdaptorTest.cxx
itkStdStreamLogOutputTest.cxx
itkThreadDefsTest.cxx
itkThreadLoggerTest.cxx
itkTimeProbesTest.cxx
itkTimeStampTest.cxx
itkTransformTest.cxx
itkTransformFactoryBaseTest.cxx
itkTransformsSetParametersTest.cxx
itkTranslationTransformTest.cxx
itkTreeContainerTest.cxx
itkTreeContainerTest2.cxx
itkTriangleCellTest.cxx
itkVarianceImageFunctionTest.cxx
itkVectorGeometryTest.cxx
itkVectorImageTest.cxx
itkVectorMeanImageFunctionTest.cxx
itkVersorTransformTest.cxx
itkVersorTest.cxx
itkVersorRigid3DTransformTest.cxx
itkVectorTest.cxx
itkVectorInterpolateImageFunctionTest.cxx
itkVectorToRGBImageAdaptorTest.cxx
itkWindowedSincInterpolateImageFunctionTest.cxx
itkEllipsoidInteriorExteriorSpatialFunctionTest.cxx
itkSymmetricEllipsoidInteriorExteriorSpatialFunctionTest.cxx
itkSymmetricSecondRankTensorTest.cxx
itkSymmetricEigenAnalysisTest.cxx
itkSTLThreadTest.cxx
itkVariableLengthVectorTest.cxx
itkVariableSizeMatrixTest.cxx
itkImportContainerTest.cxx
itkZeroFluxBoundaryConditionTest.cxx
itkImageFillBufferTest.cxx
)
# itkQuaternionOrientationAdapterTest.cxx

###
# will be moved to a common test driver after the file is determined to be portable and the rounding has been fully migrated
ADD_EXECUTABLE( itkMathTest itkMathTest.cxx )
ADD_TEST( itkMathTest  ${CXX_TEST_PATH}/itkMathTest )
###

CONFIGURE_FILE(${ITKTesting_SOURCE_DIR}/Code/Common/itkSystemInformationTest.h.in
               ${ITKTesting_BINARY_DIR}/Code/Common/itkSystemInformationTest.h
               @ONLY IMMEDIATE)
INCLUDE_DIRECTORIES("${ITKTesting_BINARY_DIR}")

IF(ITK_OBJCXX_COMPILER_WORKS)
  SET(COMMON_OBJCXX_TEST ${CXX_TEST_PATH}/itkCommonHeaderObjCxxTest)
  ADD_TEST(itkCommonHeaderObjCxxTest ${COMMON_OBJCXX_TEST})
  ADD_EXECUTABLE(itkCommonHeaderObjCxxTest itkCommonHeaderObjCxxTest.mm)
  TARGET_LINK_LIBRARIES(itkCommonHeaderObjCxxTest ITKCommon)
ENDIF(ITK_OBJCXX_COMPILER_WORKS)

ADD_EXECUTABLE(itkSystemInformationTest itkSystemInformationTest.cxx)
ADD_EXECUTABLE(itkSystemInformation itkSystemInformation.cxx)
ADD_EXECUTABLE(itkCommonHeaderTest itkCommonHeaderTest.cxx)
ADD_EXECUTABLE(itkCommonTests itkCommonTests.cxx ${BasicTests_SRCS})
ADD_EXECUTABLE(itkCommonTests2 itkCommonTests2.cxx ${BasicTests2_SRCS})
TARGET_LINK_LIBRARIES(itkCommonHeaderTest ITKCommon)
TARGET_LINK_LIBRARIES(itkCommonTests ITKIO)
TARGET_LINK_LIBRARIES(itkCommonTests2 ITKIO)
SET_TARGET_PROPERTIES(itkCommonTests2 PROPERTIES ENABLE_EXPORTS 1)
TARGET_LINK_LIBRARIES(itkSystemInformation itksys)

IF(ITK_BUILD_SHARED_LIBS)
## Create a library to test ITK loadable factories
SET(FactoryTest_Source itkFactoryTestLib.cxx)
SET(LIBRARY_OUTPUT_PATH "${ITKTesting_BINARY_DIR}/Temporary")
ADD_LIBRARY(FactoryTestLib SHARED ${FactoryTest_Source})
TARGET_LINK_LIBRARIES (FactoryTestLib ITKCommon)
ADD_DEPENDENCIES(itkCommonTests2 FactoryTestLib)
ENDIF(ITK_BUILD_SHARED_LIBS)
ENDIF( NOT ITK_DISABLE_CXX_TESTING )


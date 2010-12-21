/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkBinaryThresholdSpatialFunctionTest.cxx,v $
  Language:  C++
  Date:      $Date: 2008-12-07 13:03:14 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkBinaryThresholdSpatialFunction.h"
#include "itkSphereSignedDistanceFunction.h"

#include "itkFloodFilledSpatialFunctionConditionalConstIterator.h"
#include "itkImage.h"

/**
 * This module tests the sphereality of the
 * BinaryThresholdSpatialFunction class.
 *
 * In particular, it creates a SphereSignedDistanceFunction object
 * connect it to a BinaryThresholdSpatialFunction class.
 *
 * The sphere parameters are set at radius of 5 and center of (0,0).
 * Memebership (i.e. with user specified threshold) is evaluated at 
 * several point and compared to expected values. 
 * The test fails if the evaluated results is not the same as expected
 * results.
 *
 */
int itkBinaryThresholdSpatialFunctionTest( int, char *[])
{
  typedef double CoordRep;
  const unsigned int Dimension = 2;
 
  typedef itk::SphereSignedDistanceFunction<CoordRep,Dimension> SphereFunctionType;
  typedef itk::BinaryThresholdSpatialFunction<SphereFunctionType> FunctionType;
  typedef SphereFunctionType::PointType PointType;
  typedef SphereFunctionType::ParametersType ParametersType;

  SphereFunctionType::Pointer sphere = SphereFunctionType::New();

  // we must initialize the sphere before use
  sphere->Initialize();

  ParametersType parameters( sphere->GetNumberOfParameters() );
  parameters.Fill( 0.0 );
  parameters[0] = 5.0;

  sphere->SetParameters( parameters );

  std::cout << "SphereParameters: " << sphere->GetParameters() << std::endl;

  // create a binary threshold function 
  FunctionType::Pointer function = FunctionType::New();

  // connect the sphere function
  function->SetFunction( sphere );

  // set the thresholds
  double lowerThreshold = -3.0;
  double upperThreshold =  4.0;
  function->SetLowerThreshold( lowerThreshold );
  function->SetUpperThreshold( upperThreshold );

  std::cout << "LowerThreshold: " << function->GetLowerThreshold() << std::endl;
  std::cout << "UpperThreshold: " << function->GetUpperThreshold() << std::endl;

  PointType point;

  for ( double p = 0.0; p < 10.0; p += 1.0 )
    {
    point.Fill( p );
    FunctionType::OutputType output = function->Evaluate( point );
    std::cout << "f( " << point << ") = " << output;
    std::cout << " [" << function->GetFunction()->Evaluate( point );
    std::cout << "] " << std::endl;

    // check results
    CoordRep val = p * vcl_sqrt( 2.0 ) - parameters[0];
    bool expected = ( lowerThreshold <= val && upperThreshold >= val );
    if( output != expected )
      {
      std::cout << "But expected value is: " << expected << std::endl;
      return EXIT_FAILURE;
     }
    }

  /**
   * In the following, we demsonstrate how BinaryThresholdSpatialFunction
   * can be used to iterate over pixels whose signed distance is
   * within [lowerThreshold,upperThreshold] of the zero level set defining
   * the sphere.
   */
  // set up a dummy image
  typedef itk::Image<unsigned char,Dimension> ImageType;
  ImageType::Pointer image = ImageType::New();
  ImageType::SizeType size;
  size.Fill( 10 );
  image->SetRegions( size );
  image->Allocate();
  image->FillBuffer( 255 );

  // set up the conditional iterator
  typedef itk::FloodFilledSpatialFunctionConditionalConstIterator<
                                          ImageType,
                                          FunctionType> IteratorType;

  IteratorType iterator( image, function );
  iterator.SetOriginInclusionStrategy();

  // add a seed that already inside the region
  ImageType::IndexType index;
  index[0] = 0; index[1] = 3;
  iterator.AddSeed( index );

  unsigned int counter = 0;
  iterator.GoToBegin();
  
  while( !iterator.IsAtEnd() )
    {

    index = iterator.GetIndex();
    image->TransformIndexToPhysicalPoint( index, point );
    double value = sphere->Evaluate( point );

    std::cout << counter++ << ": ";
    std::cout << index << " ";
    std::cout << value << " ";
    std::cout << std::endl;

    // check if value is within range
    if ( value < lowerThreshold || value > upperThreshold )
      {
      std::cout << "Point value is not within thresholds [";
      std::cout << lowerThreshold << "," << upperThreshold << "]" << std::endl;
      return EXIT_FAILURE;
      }

    ++iterator;
    };


  function->Print(std::cout);

  std::cout << "Test passed." << std::endl;
  return EXIT_SUCCESS;
}


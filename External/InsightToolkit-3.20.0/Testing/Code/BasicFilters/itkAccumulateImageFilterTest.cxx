/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAccumulateImageFilterTest.cxx,v $
  Language:  C++
  Date:      $Date: 2006-02-25 16:50:50 $
  Version:   $Revision: 1.4 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkAccumulateImageFilter.h"
#include "itkRGBPixel.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itksys/SystemTools.hxx"

int itkAccumulateImageFilterTest(int argc, char *argv[] )
{
  typedef short PixelType;
  static const int ImageDimension = 3;

  typedef itk::Image<PixelType,ImageDimension> InputImageType;
  typedef itk::Image<PixelType,ImageDimension> OutputImageType;
  typedef itk::Image<unsigned char,ImageDimension> WriteImageType;
  typedef itk::ImageSeriesReader< InputImageType > ReaderType ;
  typedef itk::AccumulateImageFilter<InputImageType,OutputImageType> AccumulaterType;
  typedef itk::ImageSeriesWriter<OutputImageType,WriteImageType> WriterType; 
  typedef itk::GDCMSeriesFileNames                SeriesFileNames;
  typedef itk::GDCMImageIO                        ImageIOType;

  if (argc < 3)
    {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  inputDICOMDirectory outputFile" << std::endl;
    return EXIT_FAILURE;
    }

  // Get the input filenames
  SeriesFileNames::Pointer names = SeriesFileNames::New();

  // Get the DICOM filenames from the directory
  names->SetInputDirectory( argv[1] );

  // Create the reader
  ImageIOType::Pointer gdcmIO = ImageIOType::New();
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO( gdcmIO );
  try
    {
    reader->SetFileNames( names->GetInputFileNames() );
    reader->Update();
    }
  catch (itk::ExceptionObject &excp)
    {
    std::cerr << "Error reading the series" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  // Accumulate the input images
  AccumulaterType::Pointer accumulate = AccumulaterType::New();
  accumulate->SetInput( reader->GetOutput() );
  accumulate->SetAccumulateDimension( 2 );

  try
    {
    accumulate->Update();
    }
  catch ( itk::ExceptionObject &excp)
    {
    std::cerr << "Error running the accumulate filter" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }
  accumulate->GetOutput()->Print(std::cout);

  accumulate->Print( std::cout );

  // Now turn averaging off
  accumulate->AverageOff();
  try
    {
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName ( argv[2] );
    
    writer->SetInput(accumulate->GetOutput());
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Error writing the series" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;

    }

  // Now turn averaging on
  accumulate->AverageOn();
  std::cout << "Average: " << accumulate->GetAverage() << std::endl;

  try
    {
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName ( argv[2] );
    
    writer->SetInput(accumulate->GetOutput());
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << "Error writing the series" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;

    }

  // Test dimension check exception.
  try
    {
    accumulate->SetAccumulateDimension( 5 );
    accumulate->Update();
    std::cout << "Failed to catch expected exception." << std::endl;
    return EXIT_FAILURE;
    }
  catch ( itk::ExceptionObject &excp )
    {
    std::cout << "Caught expected exception." << std::endl;
    std::cout << excp << std::endl;
    }
  std::cout << "Test passed." << std::endl;
  return EXIT_SUCCESS;

}

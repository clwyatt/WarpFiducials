/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkImageRandomIteratorTest.cxx,v $
  Language:  C++
  Date:      $Date: 2008-01-18 18:24:13 $
  Version:   $Revision: 1.10 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRandomIteratorWithIndex.h"
#include "itkImageRandomConstIteratorWithIndex.h"




int itkImageRandomIteratorTest(int, char* [] )
{
  std::cout << "Creating an image of indices" << std::endl;

  const unsigned int ImageDimension = 3;

  typedef itk::Index< ImageDimension >             PixelType;

  typedef itk::Image< PixelType, ImageDimension >  ImageType;

  ImageType::Pointer myImage = ImageType::New();
  ImageType::ConstPointer myConstImage = myImage.GetPointer();
  
  ImageType::SizeType size0;

  size0[0] = 100;
  size0[1] = 100;
  size0[2] = 100;

  unsigned long numberOfSamples = 10;

  ImageType::IndexType start0;
  start0.Fill(0);

  ImageType::RegionType region0;
  region0.SetIndex( start0 );
  region0.SetSize( size0 );

  myImage->SetLargestPossibleRegion( region0 );
  myImage->SetBufferedRegion( region0 );
  myImage->SetRequestedRegion( region0 );
  myImage->Allocate();

  typedef itk::ImageRegionIteratorWithIndex< ImageType >            IteratorType;

  typedef itk::ImageRandomIteratorWithIndex< ImageType >      RandomIteratorType;

  typedef itk::ImageRandomConstIteratorWithIndex< ImageType > RandomConstIteratorType;

  IteratorType it( myImage, region0 );

  it.GoToBegin();
  ImageType::IndexType index0;
  
  // Fill an image with indices
  while( !it.IsAtEnd() )
  {
    index0 = it.GetIndex();
    it.Set( index0 );
    ++it;
  }

  
  // Sample the image 
  RandomIteratorType ot( myImage, region0 );
  ot.SetNumberOfSamples( numberOfSamples ); 
  ot.GoToBegin();

 
  std::cout << "Verifying non-const iterator... ";
  std::cout << "Random walk of the Iterator over the image " << std::endl;
  while( !ot.IsAtEnd() )
    {
    index0 = ot.GetIndex();
    if( ot.Get() != index0 )
      {
        std::cerr << "Values don't correspond to what was stored "
          << std::endl;
        std::cerr << "Test failed at index ";
        std::cerr << index0 << std::endl;
        return EXIT_FAILURE;
      }
    std::cout << index0 << std::endl;
    ++ot;
    }
  std::cout << "   Done ! " << std::endl;

  
  // Verification 
  RandomConstIteratorType cot( myConstImage, region0 );
  cot.SetNumberOfSamples( numberOfSamples );
  cot.GoToBegin();

 
  std::cout << "Verifying const iterator... ";
  std::cout << "Random walk of the Iterator over the image " << std::endl;

  while( !cot.IsAtEnd() )
  {
    index0 = cot.GetIndex();
    if( cot.Get() != index0 )
      {
      std::cerr << "Values don't correspond to what was stored "
        << std::endl;
      std::cerr << "Test failed at index ";
      std::cerr << index0 << " value is " << cot.Get() <<  std::endl;
      return EXIT_FAILURE;
      }
    std::cout << index0 << std::endl;
    ++cot;
  }
  std::cout << "   Done ! " << std::endl;



  // Verification 
  std::cout << "Verifying iterator in reverse direction... " << std::endl;
  std::cout << "Should be a random walk too (a different one)" << std::endl;

  RandomIteratorType ior( myImage, region0 );
  ior.SetNumberOfSamples( numberOfSamples );
  ior.GoToEnd();

  --ior;
 

  while( !ior.IsAtBegin() )
  {
    index0 = ior.GetIndex();
    if( ior.Get() != index0 )
    {
      std::cerr << "Values don't correspond to what was stored "
        << std::endl;
      std::cerr << "Test failed at index ";
      std::cerr << index0 << " value is " << ior.Get() <<  std::endl;
      return EXIT_FAILURE;
    }
    std::cout << index0 << std::endl;
    --ior;
  }
  std::cout << index0 << std::endl; // print the value at the beginning index
  std::cout << "   Done ! " << std::endl;



  // Verification 
  std::cout << "Verifying const iterator in reverse direction... ";

  RandomConstIteratorType cor( myImage, region0 );
  cor.SetNumberOfSamples( numberOfSamples ); // 0=x, 1=y, 2=z
  cor.GoToEnd();

  --cor; // start at the end position 

  while( !cor.IsAtBegin() )
    {
    index0 = cor.GetIndex();
    if( cor.Get() != index0 )
      {
      std::cerr << "Values don't correspond to what was stored "
        << std::endl;
      std::cerr << "Test failed at index ";
      std::cerr << index0 << " value is " << cor.Get() <<  std::endl;
      return EXIT_FAILURE;
      }
    std::cout << index0 << std::endl;
    --cor;
    }
  std::cout << index0 << std::endl; // print the value at the beginning index
  std::cout << "   Done ! " << std::endl;

 // Verification 
  std::cout << "Verifying const iterator in both directions... ";

  RandomConstIteratorType dor( myImage, region0 );
  dor.SetNumberOfSamples( numberOfSamples ); // 0=x, 1=y, 2=z
  dor.GoToEnd();

  --dor; // start at the last valid pixel position 

  for (unsigned int counter = 0; ! dor.IsAtEnd(); ++counter)
    {
      index0 = dor.GetIndex();
      if( dor.Get() != index0 )
        {
          std::cerr << "Values don't correspond to what was stored "
                    << std::endl;
          std::cerr << "Test failed at index ";
          std::cerr << index0 << " value is " << dor.Get() <<  std::endl;
          return EXIT_FAILURE;
        }
      std::cout << index0 << std::endl;
      if (counter < 6)  { --dor; }
      else { ++dor; }
    }
  std::cout << index0 << std::endl; // print the value at the beginning index
  std::cout << "   Done ! " << std::endl;
  

  // Verification of the Iterator in a subregion of the image
  {
    std::cout << "Verifying Iterator in a Region smaller than the whole image... "
              << std::endl;

    ImageType::IndexType start;
    start[0] = 10;
    start[1] = 12;
    start[2] = 14;
    
    ImageType::SizeType size;
    size[0] = 11;
    size[1] = 12;
    size[2] = 13;

    ImageType::RegionType region;
    region.SetIndex( start );
    region.SetSize( size );

    RandomIteratorType cbot( myImage, region );

    cbot.SetNumberOfSamples( numberOfSamples ); // 0=x, 1=y, 2=z
    cbot.GoToBegin();

    while( !cbot.IsAtEnd() )
      {
      ImageType::IndexType index =  cbot.GetIndex();
      ImageType::PixelType pixel =  cbot.Get();

      if( index != pixel )
        {
        std::cerr << "Iterator in region test failed" << std::endl;
        std::cerr << pixel << " should be" << index << std::endl;
        return EXIT_FAILURE;
        }

      if( !region.IsInside( index ) )
        {
        std::cerr << "Iterator in region test failed" << std::endl;
        std::cerr << index << " is outside the region " << region << std::endl;
        return EXIT_FAILURE;
        }
      std::cout << index << std::endl;
      ++cbot;
      }

    std::cout << "   Done ! " << std::endl;
  }



  // Verification of the Const Iterator in a subregion of the image
  {
    std::cout << "Verifying Const Iterator in a Region smaller than the whole image... "
              << std::endl;

    ImageType::IndexType start;
    start[0] = 10;
    start[1] = 12;
    start[2] = 14;
    
    ImageType::SizeType size;
    size[0] = 11;
    size[1] = 12;
    size[2] = 13;

    ImageType::RegionType region;
    region.SetIndex( start );
    region.SetSize( size );

    RandomConstIteratorType cbot( myImage, region );

    cbot.SetNumberOfSamples( numberOfSamples );
    cbot.GoToBegin();

    while( !cbot.IsAtEnd() )
      {
      ImageType::IndexType index =  cbot.GetIndex();
      ImageType::PixelType pixel =  cbot.Get();

      if( index != pixel )
        {
        std::cerr << "Iterator in region test failed" << std::endl;
        std::cerr << pixel << " should be" << index << std::endl;
        return EXIT_FAILURE;
        }
      if( !region.IsInside( index ) )
        {
        std::cerr << "Iterator in region test failed" << std::endl;
        std::cerr << index << " is outside the region " << region << std::endl;
        return EXIT_FAILURE;
        }
      std::cout << index << std::endl;

      ++cbot;
      }

    std::cout << "   Done ! " << std::endl;
  }


  std::cout << "Test passed" << std::endl;




    return EXIT_SUCCESS;

  }




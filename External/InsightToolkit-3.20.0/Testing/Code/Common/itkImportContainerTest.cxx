/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkImportContainerTest.cxx,v $
  Language:  C++
  Date:      $Date: 2008-01-24 13:42:33 $
  Version:   $Revision: 1.7 $

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
#include "itkImportImageContainer.h"
#include "itkNumericTraits.h"
#include "itkOutputWindow.h"
#include "itkTextOutput.h"

int itkImportContainerTest(int , char * [] )
{
  typedef float PixelType;
  typedef itk::ImportImageContainer<unsigned long, PixelType> ContainerType;

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

// First test with ContainerManagesMemory false
  PixelType *ptr1;
  {
  // Test 1: Create an empty container and print it
  ContainerType::Pointer container1 = ContainerType::New();
  container1->Print(std::cout);
  std::cout << "After New(), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;


  // Test 2: Reserve memory
  container1->Reserve(1000);
  std::cout << "After container1->Reserve(1000), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  // Take control of the pointer
  container1->ContainerManageMemoryOff();
  ptr1 = container1->GetImportPointer();

  // Test 3: Reserve a smaller amount of memory
  container1->Reserve(100);
  std::cout << "After container1->Reserve(100), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;


  // Test 4: Squeeze the container
  container1->Squeeze();
  std::cout << "After container1->Squeeze() size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  // Test 5: Initialize the container
  container1->Initialize();
  std::cout << "After container1->Initialize() size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;
  }

// now repeat the tests with a user provided pointer
  PixelType *myPtr = new float[2000];
  {
  // Test 1: Create an empty container and print it
  ContainerType::Pointer container1 = ContainerType::New();
  container1->Print(std::cout);
  std::cout << "After New(), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  container1->SetImportPointer(myPtr, 2000);
  std::cout << "After New(), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  *(myPtr + 500) = 500.0;

  // Test 2: Reserve less memory than capacity
  container1->Reserve(1000);
  std::cout << "After container1->Reserve(1000), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  if (*(myPtr + 500) != 500.0)
    {
    std::cout << "Test failed: After container1->Reserve(1000), container1[500] does != 500." << std::endl;
    return EXIT_FAILURE;
    }

  // Test 3: Reserve more memory than capacity
  container1->Reserve(10000);
  std::cout << "After container1->Reserve(100), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;
  if (*(myPtr + 500) != 500.0)
    {
    std::cout << "Test failed: After container1->Reserve(10000), container1[500] does != 500." << std::endl;
    return EXIT_FAILURE;
    }


  // Test 4: Squeeze the container
  container1->Squeeze();
  std::cout << "After container1->Squeeze() size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  if (*(myPtr + 500) != 500.0)
    {
    std::cout << "Test failed: After container1->Squeeze(), container1[500] does != 500." << std::endl;
    return EXIT_FAILURE;
    }

  // Test 5: Initialize the container
  container1->Initialize();
  std::cout << "After container1->Initialize() size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;
  }

  // Now repeat tests with ContainerManagesMemory true
  {
  // Test 1: Create an empty container and print it
  ContainerType::Pointer container1 = ContainerType::New();
  container1->Print(std::cout);
  std::cout << "After New(), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;


  // Test 2: Reserve memory
  container1->Reserve(1000);
  std::cout << "After container1->Reserve(1000), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  // Test 3: Reserve a smaller amount of memory
  container1->Reserve(100);
  std::cout << "After container1->Reserve(100), size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  // Test 4: Squeeze the container
  container1->Squeeze();
  std::cout << "After container1->Squeeze() size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;

  // Test 5: Initialize the container
  container1->Initialize();
  std::cout << "After container1->Initialize() size is "
            << container1->Size()
            << ", capacity is "
            <<  container1->Capacity()
            << " and import pointer is "
            << container1->GetImportPointer()
            << std::endl;
  }

  // valgrind has problems with exceptions after a failed memory
  // allocation. Since valgrind is normally built with debug, a check
  // for NDEBUG will eliminate this code. Unfortunately, coverage is
  // also built debug, so the coverage report will show that the
  // exception after a refused allocation is not exercised.
  // SGI's always say they can get you the memory you want.
#if (!defined(__sgi) && defined(NDEBUG))
  // Check the exception on the memory allocation
  bool caughtException = false;
  try
    {
    ContainerType::Pointer container1 = ContainerType::New();
    container1->Reserve( itk::NumericTraits<unsigned long>::max()/sizeof(PixelType));
    }
  catch (itk::ExceptionObject& err)
    {
    std::cout << "Caught expected exception: " << err << std::endl;
    caughtException = true;
    }
#endif

  // We must delete the memory we said we would manage
  delete [] ptr1;
  delete [] myPtr;

  // SGI's always say they can get you the memory you want.
#if (!defined(__sgi) && defined(NDEBUG))
  if (!caughtException && (sizeof(void *) != 8))
    {
    std::cout << "Failed to catch expected exception" << std::endl;
    return EXIT_FAILURE;
    }
#endif
  return EXIT_SUCCESS;
}

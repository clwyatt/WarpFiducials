/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkBioCellularAggregateTest.cxx,v $
  Language:  C++
  Date:      $Date: 2004-11-16 15:35:27 $
  Version:   $Revision: 1.3 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#pragma warning ( disable : 4503 )
#endif

#include <iostream>

#include "itkBioCellularAggregate.h"


int itkBioCellularAggregateTest( int, char * [] )
{
   typedef itk::bio::CellularAggregate<4> CellularAggregateType;
   
   CellularAggregateType::Pointer  aggregate = CellularAggregateType::New();


   std::cout << "Test Passed !" << std::endl;
   return EXIT_SUCCESS;
}












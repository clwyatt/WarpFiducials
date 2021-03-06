/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: testMetaLandmark.cxx,v $
  Language:  C++
  Date:      $Date: 2009-11-13 17:51:20 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <stdio.h>
#include <ctype.h>
#include <cstdlib>
#include <metaLandmark.h>
#include <itksys/SystemTools.hxx>

int testMetaLandmark(int argc, char * argv[])
{
  if (argc > 1)
    {
    itksys::SystemTools::ChangeDirectory(argv[1]);
    }

  std::cout << "Creating test file ..." << std::endl;
  MetaLandmark Landmark0;
  MetaLandmark Landmark(3);
  Landmark.ID(0);
  LandmarkPnt* pnt;

  std::cout << "Allocating points..." << std::endl;
  unsigned int i;
  for(i=0;i<10;i++)
  {
    pnt = new LandmarkPnt(3);
    pnt->m_X[0]=(float)0.2;
    pnt->m_X[1]=static_cast<float>(i);
    pnt->m_X[2]=static_cast<float>(i);
    Landmark.GetPoints().push_back(pnt);
  }
  
  std::cout << "Writing test file ..." << std::endl;
   
  Landmark.BinaryData(true);
  Landmark.ElementType(MET_FLOAT);
  Landmark.Write("Landmarks.meta");

  std::cout << "  done" << std::endl;
 
  std::cout << "Reading test file ..." << std::endl;
  Landmark.Read("Landmarks.meta"); 
  MetaLandmark landmarkRead("Landmarks.meta"); 
  MetaLandmark landmarkCopy(&landmarkRead);

  std::cout << "PointDim = " << landmarkCopy.PointDim() << std::endl;
  std::cout << "NPoints = "  << landmarkCopy.NPoints() << std::endl;
  std::cout << "ElementType = " << landmarkCopy.ElementType() << std::endl;

  std::cout << "  done" << std::endl;

  Landmark.PrintInfo();

  std::cout << "Accessing pointlist..." << std::endl;

  MetaLandmark::PointListType plist =  Landmark.GetPoints();
  MetaLandmark::PointListType::const_iterator it = plist.begin();
  
  while(it != plist.end())
  {
    for(unsigned int d = 0; d < 3; d++)
    {
      std::cout << (*it)->m_X[d] << " ";
    }

    std::cout << std::endl;
    it++;
  }

  std::cout << "done" << std::endl;
  return EXIT_SUCCESS;
}

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkVectorTest.cxx,v $
  Language:  C++
  Date:      $Date: 2009-04-05 10:56:56 $
  Version:   $Revision: 1.17 $

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
#include "itkVector.h"

// Define floating point type to use for the test.
typedef double Real;

bool different(Real a, Real b)
{
  return vcl_fabs(a-b) > 1e-6;
}

int itkVectorTest(int, char* [] )
{
  bool passed = true;

  typedef itk::Vector<Real, 2> RealVector;

  RealVector s;
  RealVector t;
  RealVector r;

  int i = 4;
  Real f = 2.1;
  
  s.Fill(3.0);
  if (different(s[0], 3.0) || different(s[1], 3.0))
    {
    passed = false;
    }

  s.Fill(2);
  if (different(s[0], 2.0) || different(s[1], 2.0))
    {
    passed = false;
    }
  
  s.Fill(i);
  if (different(s[0], i) || different(s[1], i))
    {
    passed = false;
    }

  s.Fill(f);
  if (different(s[0], f) || different(s[1], f))
    {
    passed = false;
    }


  t = s;
  if (different(t[0], s[0]) || different(t[1], s[1]))
    {
    passed = false;
    }
  
  s = -t;
  if (different(s[0], -t[0]) || different(s[1], -t[1]))
    {
    passed = false; 
    }
  
  s.Fill(3.0);
  s *= 2.5;
  if (different(s[0], 7.5) || different(s[1], 7.5))
    {
    passed = false;
    }

  s /= 2.0;
  if (different(s[0], 3.75) || different(s[1], 3.75))
    {
    passed = false;
    }
 

  s.Fill(3.8);
  s *= f;
  if (different(s[0], 7.98) || different(s[1], 7.98))
    {
    passed = false;
    }  

  s /= f;
  if (different(s[0], 3.8) || different(s[1], 3.8))
    {
    passed = false;
    }

  s += t;
  if (different(s[0], 5.9) || different(s[1], 5.9))
    {
    passed = false;
    }
  

  s -= t;
  if (different(s[0], 3.8) || different(s[1], 3.8))
    {
    passed = false;
    }


  r = s + t;
  if (different(r[0], 5.9) || different(r[1], 5.9))
    {
    passed = false;
    }
  
  r = s - t;
  if (different(r[0], 1.7) || different(r[1], 1.7))
    {
    passed = false;
    }


  r = s * 10.0;
  if (different(r[0], 38.0) || different(r[1], 38.0))
    {
    passed = false;
    }

  r = s / 10.0;
  if (different(r[0], .38) || different(r[1], .38))
    {
    passed = false;
    }

  r = s * f;
  if (different(r[0], 7.98) || different(r[1], 7.98))
    {
    passed = false;
    }

  f = 2.0;
  r = s / f;
  if (different(r[0], 1.9) || different(r[1], 1.9))
    {
    passed = false;
    }

  r[1] = 7.0;
  if (different(r[0], 1.9) || different(r[1], 7.0))
    {
    passed = false;
    }

  // Test operator=
  s.Fill(10.0);
  t = s;
  if ( s != t )
    {
    passed = false;
    }
  t.Fill(20.0);
  if ( s == t )
    {
    passed = false;
    }


  typedef itk::Vector<float, 3> RealVector3;
  RealVector3 a, b, c;
  a[0] = 1.0; a[1] = 0.0; a[2] = 0.0;
  b[0] = 0.0; b[1] = 1.0; b[2] = 0.0;
  c = itk::CrossProduct(a,b);
  std::cout << "(" << a << ") cross (" << b << ") : (" << c << ")" << std::endl;

  typedef itk::Vector<double, 3> DoubleVector3;
  DoubleVector3 aa, bb, cc;
  aa[0] = 1.0; aa[1] = 0.0; aa[2] = 0.0;
  bb[0] = 0.0; bb[1] = 1.0; bb[2] = 0.0;
  cc = itk::CrossProduct(aa,bb);
  std::cout << "(" << aa << ") cross (" << bb << ") : (" << cc << ")" << std::endl;
  typedef itk::Vector<int, 3> IntVector3;
  DoubleVector3 ia, ib, ic;
  ia[0] = 1; ia[1] = 0; ia[2] = 0;
  ib[0] = 0; ib[1] = 1; ib[2] = 0;
  ic = itk::CrossProduct(ia,ib);
  std::cout << "(" << ia << ") cross (" << ib << ") : (" << ic << ")" << std::endl;
  if (passed)
    {
    std::cout << "Vector test passed." << std::endl;
    return EXIT_SUCCESS;
    }
  else
    {
    std::cout << "Vector test failed." << std::endl;
    return EXIT_FAILURE;
    }
  


}

/*****************************************************************************
Copyright (c) 2008, Bioimaging Systems Lab, Virginia Tech
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Virgina Tech nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*******************************************************************************/
#include <cstdlib>
#include <iostream>
using std::cout; using std::endl;
#include <fstream>
using std::ofstream;
#include <string>
using std::string;

// ITK
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkPoint.h>

// itksys for portable path seperator
#define itksys_SHARED_FORWARD_DIR_BUILD ""
#define itksys_SHARED_FORWARD_PATH_BUILD ""
#define itksys_SHARED_FORWARD_PATH_INSTALL ""
#define itksys_SHARED_FORWARD_EXE_BUILD ""
#define itksys_SHARED_FORWARD_EXE_INSTALL ""
#include <itksys/SharedForward.h>

// See http://hdl.handle.net/10380/3060
#include <itkSymmetricLogDomainDemonsRegistrationFilter.h>
#include <itkMultiResolutionLogDomainDeformableRegistration.h>
#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>

#include "WarpFiducialsCLP.h"

// typedef vector<Fiducial> FiducialType;
typedef float PixelType;
typedef itk::Image<PixelType,3> ImageType;
typedef itk::Vector< PixelType, 3 > VectorPixelType;
typedef itk::Image< VectorPixelType, 3 > VelocityFieldType;
typedef itk::Image< VectorPixelType, 3 > DeformationFieldType;
typedef itk::Point< float, 3 > PointType;

int main(int argc, char *argv[])
{
  // CLP magic
  PARSE_ARGS;

  // read source and target
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer source_reader = ReaderType::New();
  source_reader->SetFileName( sourceVolume.c_str() );
  ReaderType::Pointer target_reader = ReaderType::New();
  target_reader->SetFileName( targetVolume.c_str() );

  try
    {
    source_reader->Update();
    target_reader->Update();
    }
  catch(itk::ExceptionObject &error)
    {
    cout << "Error reading input images." << endl;
    cout << error << endl;
    return EXIT_FAILURE;
    }

  std::vector< std::vector<float> > outfids( infids.size() );
  for(unsigned int i = 0; i < outfids.size(); ++i)
    {
    std::vector<float> f(3);
    f[0] = infids[i][0];
    f[1] = infids[i][1];
    f[2] = infids[i][2];
    outfids[i] = f;
    }

  // save warped fiducials csv file
  string filename = outfidpath + KWSYS_SHARED_FORWARD_PATH_SEP + outfidfile;
  ofstream outfile( filename.c_str() );
  if( outfile.fail() )
    {
    cout << "Error: could not open output for writing." << endl;
    return EXIT_FAILURE;
    }

  for(unsigned int i = 0; i < outfids.size(); ++i)
    {
    float px = outfids[i][0];
    float py = outfids[i][1];
    float pz = outfids[i][2];

    outfile << "warp-P" << i << ","
	   << px << "," << py << "," << pz << ",1,1" << endl;
    }
  outfile.close();

//   // read fiducials csv file
//   FiducialType infid;
//   if( readfid(inputfidfile, infid) )
//     {
//     cout << "Error reading input fiducial file. Halting." << endl;
//     return EXIT_FAILURE;
//     }

//   // register source to target
//   typedef itk::SymmetricLogDomainDemonsRegistrationFilter
//     < ImageType, ImageType, VelocityFieldType>
//     RegistrationFilterType;
//   typedef RegistrationFilterType::GradientType GradientType;

//   RegistrationFilterType::Pointer filter
//     = RegistrationFilterType::New();

//   filter->SetMaximumUpdateStepLength( 1.0 );
//   filter->SetUseGradientType( static_cast<GradientType>(0) );
//   filter->SetNumberOfBCHApproximationTerms( 2 );
//   filter->SmoothVelocityFieldOn();
//   filter->SetStandardDeviations( 0.6 );
//   // filter->SmoothUpdateFieldOn();
//   // filter->SetUpdateFieldStandardDeviations( 1.0 );

//   typedef itk::MultiResolutionLogDomainDeformableRegistration
//     < ImageType, ImageType, VelocityFieldType, PixelType > MultiResRegistrationFilterType;
//   MultiResRegistrationFilterType::Pointer multires = MultiResRegistrationFilterType::New();

//   typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
//      VelocityFieldType,double> FieldInterpolatorType;

//   FieldInterpolatorType::Pointer VectorInterpolator = FieldInterpolatorType::New();

//   multires->GetFieldExpander()->SetInterpolator(VectorInterpolator);

//   multires->SetRegistrationFilter( filter );

//   vector<unsigned int> numIterations(3);
//   numIterations[0] = 15;
//   numIterations[1] = 10;
//   numIterations[2] = 5;

//   multires->SetNumberOfLevels( numIterations.size() );

//   multires->SetNumberOfIterations( &numIterations[0] );

//   multires->SetFixedImage( target_reader->GetOutput() );
//   multires->SetMovingImage( source_reader->GetOutput() );

//   // Compute the deformation field
//   try
//     {
//     multires->UpdateLargestPossibleRegion();
//     }
//   catch( itk::ExceptionObject& err )
//     {
//     std::cout << "Unexpected error." << std::endl;
//     std::cout << err << std::endl;
//     return EXIT_FAILURE;
//     }

//   cout << "Registration Complete." << endl;

//   // Final deformation field
//   DeformationFieldType::Pointer defField = multires->GetDeformationField();
// //  defField->DisconnectPipeline();

//   // Write output deformation field
//   // typedef itk::ImageFileWriter< DeformationFieldType > FieldWriterType;
//   // FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
//   // fieldWriter->SetFileName( "temp.vtk" );
//   // fieldWriter->SetInput( defField );

//   // try
//   //   {
//   //   fieldWriter->Update();
//   //   }
//   // catch( itk::ExceptionObject& err )
//   //   {
//   //   std::cout << "Unexpected error." << std::endl;
//   //   std::cout << err << std::endl;
//   //   }

//   // warp fiducials
//   FiducialType outfid = infid;
//   typedef DeformationFieldType::IndexType IndexType;

//   for(unsigned int i = 0; i < outfid.size(); ++i)
//     {
//     Fiducial p = outfid[i];
//     PointType originalPoint;
//     originalPoint[0] = -p.x; // convert RAS to LPS
//     originalPoint[1] = -p.y; // convert RAS to LPS
//     originalPoint[2] = p.z;
//     IndexType index;
//     defField->TransformPhysicalPointToIndex( originalPoint, index );
//     VectorPixelType displacement = defField->GetPixel( index );
//     cout << index << endl;
//     cout << displacement << endl;
//     p.x = p.x + displacement[0];
//     p.y = p.y + displacement[1];
//     p.z = p.z + displacement[2];
//     outfid[i] = p;
//     }

//   // save warped fiducials csv file
//   ofstream outfidfile( outputfidfile.c_str() );
//   if( outfidfile.fail() )
//     {
//     cout << "Error: could not open output for writing." << endl;
//     return EXIT_FAILURE;
//     }

//   for(unsigned int i = 0; i < outfid.size(); ++i)
//     {
//     Fiducial p = outfid[i];
//     string name = "warp-" + p.name;
//     outfidfile << name.c_str() << ","
// 	   << p.x << "," << p.y << "," << p.z << ","
// 	   << p.active << "," << p.visible << endl;
//     }
//   outfidfile.close();

  return EXIT_SUCCESS;
}

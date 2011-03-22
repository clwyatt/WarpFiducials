/*****************************************************************************
Copyright (c) 2011, Bioimaging Systems Lab, Virginia Tech
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
#include <itkLogDomainDemonsRegistrationFilter.h>
#include <itkSymmetricLogDomainDemonsRegistrationFilter.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkGridForwardWarpImageFilter.h>
#include <itkHistogramMatchingImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkMultiResolutionLogDomainDeformableRegistration.h>
#include <itkTransformFileReader.h>
#include <itkTransformToVelocityFieldSource.h>
#include <itkVectorCentralDifferenceImageFunction.h>
#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>
#include <itkWarpHarmonicEnergyCalculator.h>
#include <itkWarpImageFilter.h>
#include <itkPoint.h>

// See http://hdl.handle.net/10380/3060
#include <itkSymmetricLogDomainDemonsRegistrationFilter.h>
#include <itkMultiResolutionLogDomainDeformableRegistration.h>
#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>

#include "WarpFiducialsCLP.h"

// FIXME assumes WIN32 or UNIX only
#ifdef _WIN32
const char PATH_SEP = '\\';
#else
const char PATH_SEP = '/';
#endif

struct arguments
{
  std::string  fixedImageFile;  /* -f option */
  std::string  movingImageFile; /* -m option */
  std::string  inputFieldFile;  /* -b option */
  std::string  inputTransformFile;  /* -p option */
  std::string  outputImageFile; /* -o option */
  std::string  outputDeformationFieldFile;
  std::string  outputInverseDeformationFieldFile;
  std::string  outputVelocityFieldFile;
  std::string  trueFieldFile;   /* -r option */
  std::vector<unsigned int> numIterations;   /* -i option */
  float sigmaVel;               /* -s option */
  float sigmaUp;                /* -g option */
  float maxStepLength;          /* -l option */
  unsigned int updateRule;      /* -a option */
  unsigned int gradientType;    /* -t option */
  unsigned int NumberOfBCHApproximationTerms; /* -c option */
  bool useHistogramMatching;    /* -e option */
  unsigned int verbosity;       /* -d option */
  std::string outfidpath;
  std::string outfidfile;

  friend std::ostream& operator<< (std::ostream& o, const arguments& args)
    {
    std::ostringstream osstr;
    for (unsigned int i=0; i<args.numIterations.size(); ++i)
      {
      osstr<<args.numIterations[i]<<" ";
      }
    std::string iterstr = "[ " + osstr.str() + "]";

    std::string gtypeStr;
    switch (args.gradientType)
    {
    case 0:
      gtypeStr = "symmetrized (ESM for diffeomorphic and compositive)";
      break;
    case 1:
      gtypeStr = "fixed image (Thirion's vanilla forces)";
      break;
    case 2:
      gtypeStr = "warped moving image (Gauss-Newton for diffeomorphic and compositive)";
      break;
    case 3:
      gtypeStr = "mapped moving image (Gauss-Newton for additive)";
      break;
    default:
      gtypeStr = "unsuported";
    }

    std::string uruleStr;
    switch (args.updateRule)
    {
    case 0:
      uruleStr = "BCH approximation on velocity fields (log-domain)";
      break;
    case 1:
      uruleStr = "Symmetrized BCH approximation on velocity fields (symmetric log-domain)";
      break;
    default:
      uruleStr = "unsuported";
    }

    std::string histoMatchStr = (args.useHistogramMatching?"true":"false");

    return o
      <<"Arguments structure:"<<std::endl
      <<"  Fixed image file: "<<args.fixedImageFile<<std::endl
      <<"  Moving image file: "<<args.movingImageFile<<std::endl
      <<"  Input velocity field file: "<<args.inputFieldFile<<std::endl
      <<"  Input transform file: "<<args.inputTransformFile<<std::endl
      <<"  Output image file: "<<args.outputImageFile<<std::endl
      <<"  Output deformation field file: "<<args.outputDeformationFieldFile<<std::endl
      <<"  Output inverse deformation field file: "<<args.outputInverseDeformationFieldFile<<std::endl
      <<"  Output velocity field file: "<<args.outputVelocityFieldFile<<std::endl
      <<"  True deformation field file: "<<args.trueFieldFile<<std::endl
      <<"  Number of multiresolution levels: "<<args.numIterations.size()<<std::endl
      <<"  Number of log-domain demons iterations: "<<iterstr<<std::endl
      <<"  Velocity field sigma: "<<args.sigmaVel<<std::endl
      <<"  Update field sigma: "<<args.sigmaUp<<std::endl
      <<"  Maximum update step length: "<<args.maxStepLength<<std::endl
      <<"  Update rule: "<<uruleStr<<std::endl
      <<"  Type of gradient: "<<gtypeStr<<std::endl
      <<"  Number of terms in the BCH expansion: "<<args.NumberOfBCHApproximationTerms<<std::endl
      <<"  Use histogram matching: "<<histoMatchStr<<std::endl
      <<"  Algorithm verbosity (debug level): "<<args.verbosity;
    }
};

// The following was adapted from http://hdl.handle.net/10380/3060
template <unsigned int Dimension>
void LogDomainDemonsRegistrationFunction( arguments args, std::vector< std::vector<float> > &infids )
{
  // Declare the types of the images (float or double only)
  typedef float                               PixelType;
  typedef itk::Image< PixelType, Dimension >  ImageType;

  typedef itk::Vector< PixelType, Dimension > VectorPixelType;
  typedef typename itk::Image
   < VectorPixelType, Dimension >             VelocityFieldType;
  typedef typename itk::Image
   < VectorPixelType, Dimension >             DeformationFieldType;


  // Images we use
  typename ImageType::Pointer fixedImage = 0;
  typename ImageType::Pointer movingImage = 0;
  typename VelocityFieldType::Pointer inputVelField = 0;


  // Set up the file readers
  typedef itk::ImageFileReader< ImageType >         FixedImageReaderType;
  typedef itk::ImageFileReader< ImageType >         MovingImageReaderType;
  typedef itk::ImageFileReader< VelocityFieldType > VelocityFieldReaderType;
  typedef itk::TransformFileReader                  TransformReaderType;

  {//for mem allocations

  typename FixedImageReaderType::Pointer fixedImageReader
     = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingImageReader
     = MovingImageReaderType::New();

  fixedImageReader->SetFileName( args.fixedImageFile.c_str() );
  movingImageReader->SetFileName( args.movingImageFile.c_str() );


  // Update the reader
  try
    {
    fixedImageReader->Update();
    movingImageReader->Update();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << "Could not read one of the input images." << std::endl;
    std::cout << err << std::endl;
    exit( EXIT_FAILURE );
    }

  if ( ! args.inputFieldFile.empty() )
    {
    // Set up the file readers
    typename VelocityFieldReaderType::Pointer fieldReader = VelocityFieldReaderType::New();
    fieldReader->SetFileName(  args.inputFieldFile.c_str() );

    // Update the reader
    try
      {
      fieldReader->Update();
      }
    catch( itk::ExceptionObject& err )
      {
      std::cout << "Could not read the input field." << std::endl;
      std::cout << err << std::endl;
      exit( EXIT_FAILURE );
      }

    inputVelField = fieldReader->GetOutput();
    inputVelField->DisconnectPipeline();
    }
  else if ( ! args.inputTransformFile.empty() )
    {
    // Set up the transform reader
    typename TransformReaderType::Pointer transformReader
       = TransformReaderType::New();
    transformReader->SetFileName(  args.inputTransformFile.c_str() );

    // Update the reader
    try
      {
      transformReader->Update();
      }
    catch( itk::ExceptionObject& err )
      {
      std::cout << "Could not read the input transform." << std::endl;
      std::cout << err << std::endl;
      exit( EXIT_FAILURE );
      }

    typedef typename TransformReaderType::TransformType BaseTransformType;
    BaseTransformType* baseTrsf(0);

    const typename TransformReaderType::TransformListType* trsflistptr
       = transformReader->GetTransformList();
    if ( trsflistptr->empty() )
      {
      std::cout << "Could not read the input transform." << std::endl;
      exit( EXIT_FAILURE );
      }
    else if (trsflistptr->size()>1 )
      {
      std::cout << "The input transform file contains more than one transform." << std::endl;
      exit( EXIT_FAILURE );
      }

    baseTrsf = trsflistptr->front();
    if ( !baseTrsf )
      {
      std::cout << "Could not read the input transform." << std::endl;
      exit( EXIT_FAILURE );
      }


    // Set up the TransformToDeformationFieldFilter
    typedef itk::TransformToVelocityFieldSource
       <VelocityFieldType>                             FieldGeneratorType;
    typedef typename FieldGeneratorType::TransformType TransformType;

    TransformType* trsf = dynamic_cast<TransformType*>(baseTrsf);
    if ( !trsf )
      {
      std::cout << "Could not cast input transform to a usable transform." << std::endl;
      exit( EXIT_FAILURE );
      }

    typename FieldGeneratorType::Pointer fieldGenerator
       = FieldGeneratorType::New();

    fieldGenerator->SetTransform( trsf );
    fieldGenerator->SetOutputParametersFromImage(
       fixedImageReader->GetOutput() );

    // Update the fieldGenerator
    try
      {
      fieldGenerator->Update();
      }
    catch( itk::ExceptionObject& err )
      {
      std::cout << "Could not generate the input field." << std::endl;
      std::cout << err << std::endl;
      exit( EXIT_FAILURE );
      }

    inputVelField = fieldGenerator->GetOutput();
    inputVelField->DisconnectPipeline();
    }


  if (!args.useHistogramMatching)
    {
    fixedImage = fixedImageReader->GetOutput();
    fixedImage->DisconnectPipeline();
    movingImage = movingImageReader->GetOutput();
    movingImage->DisconnectPipeline();
    }
  else
    {
    // match intensities
    ///\todo use inputDefField if any to get a better guess?
    typedef itk::HistogramMatchingImageFilter
       <ImageType, ImageType> MatchingFilterType;
    typename MatchingFilterType::Pointer matcher = MatchingFilterType::New();

    matcher->SetInput( movingImageReader->GetOutput() );
    matcher->SetReferenceImage( fixedImageReader->GetOutput() );

    matcher->SetNumberOfHistogramLevels( 1024 );
    matcher->SetNumberOfMatchPoints( 7 );
    matcher->ThresholdAtMeanIntensityOn();

    // Update the matcher
    try
      {
      matcher->Update();
      }
    catch( itk::ExceptionObject& err )
      {
      std::cout << "Could not match the input images." << std::endl;
      std::cout << err << std::endl;
      exit( EXIT_FAILURE );
      }

    movingImage = matcher->GetOutput();
    movingImage->DisconnectPipeline();

    fixedImage = fixedImageReader->GetOutput();
    fixedImage->DisconnectPipeline();
    }

  }//end for mem allocations


  // Set up the demons filter output deformation field
  typename DeformationFieldType::Pointer defField = 0;
  typename DeformationFieldType::Pointer invDefField = 0;
  typename VelocityFieldType::Pointer velField = 0;

  {//for mem allocations

  // Set up the demons filter
  typedef typename itk::LogDomainDeformableRegistrationFilter
     < ImageType, ImageType, VelocityFieldType>   BaseRegistrationFilterType;
  typename BaseRegistrationFilterType::Pointer filter;

  switch (args.updateRule)
  {
  case 0:
    {
    // exp(v) <- exp(v) o exp(u) (log-domain demons)
    typedef typename itk::LogDomainDemonsRegistrationFilter
       < ImageType, ImageType, VelocityFieldType>
       ActualRegistrationFilterType;
    typedef typename ActualRegistrationFilterType::GradientType GradientType;

    typename ActualRegistrationFilterType::Pointer actualfilter
       = ActualRegistrationFilterType::New();

    actualfilter->SetMaximumUpdateStepLength( args.maxStepLength );
    actualfilter->SetUseGradientType(
       static_cast<GradientType>(args.gradientType) );
    actualfilter->SetNumberOfBCHApproximationTerms(args.NumberOfBCHApproximationTerms);
    filter = actualfilter;

    break;
    }
  case 1:
    {
    // exp(v) <- Symmetrized( exp(v) o exp(u) ) (symmetriclog-domain demons)
    typedef typename itk::SymmetricLogDomainDemonsRegistrationFilter
       < ImageType, ImageType, VelocityFieldType>
       ActualRegistrationFilterType;
    typedef typename ActualRegistrationFilterType::GradientType GradientType;

    typename ActualRegistrationFilterType::Pointer actualfilter
       = ActualRegistrationFilterType::New();

    actualfilter->SetMaximumUpdateStepLength( args.maxStepLength );
    actualfilter->SetUseGradientType(
       static_cast<GradientType>(args.gradientType) );
    actualfilter->SetNumberOfBCHApproximationTerms(args.NumberOfBCHApproximationTerms);
    filter = actualfilter;

    break;
    }
  default:
    {
    std::cout << "Unsupported update rule." << std::endl;
    exit( EXIT_FAILURE );
    }
  }

  if ( args.sigmaVel > 0.1 )
    {
    filter->SmoothVelocityFieldOn();
    filter->SetStandardDeviations( args.sigmaVel );
    }
  else
    {
    filter->SmoothVelocityFieldOff();
    }

  if ( args.sigmaUp > 0.1 )
    {
    filter->SmoothUpdateFieldOn();
    filter->SetUpdateFieldStandardDeviations( args.sigmaUp );
    }
  else
    {
    filter->SmoothUpdateFieldOff();
    }

  //filter->SetIntensityDifferenceThreshold( 0.001 );

  if ( args.verbosity > 0 )
    {
    if ( ! args.trueFieldFile.empty() )
      {
      if (args.numIterations.size()>1)
        {
        std::cout << "You cannot compare the results with a true field in a multiresolution setting yet." << std::endl;
        exit( EXIT_FAILURE );
        }

      // Set up the file readers
      typedef itk::ImageFileReader< DeformationFieldType > DeformationFieldReaderType;
      typename DeformationFieldReaderType::Pointer fieldReader = DeformationFieldReaderType::New();
      fieldReader->SetFileName(  args.trueFieldFile.c_str() );

      // Update the reader
      try
        {
        fieldReader->Update();
        }
      catch( itk::ExceptionObject& err )
        {
        std::cout << "Could not read the true field." << std::endl;
        std::cout << err << std::endl;
        exit( EXIT_FAILURE );
        }
      }
    }

  // Set up the multi-resolution filter
  typedef typename itk::MultiResolutionLogDomainDeformableRegistration<
     ImageType, ImageType, VelocityFieldType, PixelType >   MultiResRegistrationFilterType;
  typename MultiResRegistrationFilterType::Pointer multires = MultiResRegistrationFilterType::New();

  typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
     VelocityFieldType,double> FieldInterpolatorType;

  typename FieldInterpolatorType::Pointer VectorInterpolator =
     FieldInterpolatorType::New();

  multires->GetFieldExpander()->SetInterpolator(VectorInterpolator);

  multires->SetRegistrationFilter( filter );
  multires->SetNumberOfLevels( args.numIterations.size() );

  multires->SetNumberOfIterations( &args.numIterations[0] );

  multires->SetFixedImage( fixedImage );
  multires->SetMovingImage( movingImage );
  multires->SetArbitraryInitialVelocityField( inputVelField );

  // Compute the deformation field
  try
    {
    //std::cout << filter << std::endl;

    multires->UpdateLargestPossibleRegion();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << "Unexpected error." << std::endl;
    std::cout << err << std::endl;
    exit( EXIT_FAILURE );
    }


  std::cout << "Done." << std::endl;

  // Get various outputs

  // Final deformation field
  defField = multires->GetDeformationField();
  defField->DisconnectPipeline();

  // Inverse final deformation field
  invDefField = multires->GetInverseDeformationField();
  invDefField->DisconnectPipeline();

  // Final velocity field
  velField =  multires->GetVelocityField();
  velField->DisconnectPipeline();

  }//end for mem allocations

   // warp the result
  typedef itk::WarpImageFilter
    < ImageType, ImageType, DeformationFieldType >  WarperType;
  typename WarperType::Pointer warper = WarperType::New();
  warper->SetInput( movingImage );
  warper->SetOutputSpacing( fixedImage->GetSpacing() );
  warper->SetOutputOrigin( fixedImage->GetOrigin() );
  warper->SetOutputDirection( fixedImage->GetDirection() );
  warper->SetDeformationField( defField );


  // Write warped image out to file
  typedef PixelType                                OutputPixelType;
	typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::CastImageFilter
   < ImageType, OutputImageType >                  CastFilterType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  typename WriterType::Pointer      writer =  WriterType::New();
  typename CastFilterType::Pointer  caster =  CastFilterType::New();
  writer->SetFileName( args.outputImageFile.c_str() );
  caster->SetInput( warper->GetOutput() );
  writer->SetInput( caster->GetOutput()   );
  writer->SetUseCompression( true );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << "Unexpected error." << std::endl;
    std::cout << err << std::endl;
    exit( EXIT_FAILURE );
    }


  // Write output deformation field
  if (!args.outputDeformationFieldFile.empty())
    {
    // Write the deformation field as an image of vectors.
    // Note that the file format used for writing the deformation field must be
    // capable of representing multiple components per pixel. This is the case
    // for the MetaImage and VTK file formats for example.
    typedef itk::ImageFileWriter< DeformationFieldType > FieldWriterType;
    typename FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
    fieldWriter->SetFileName(  args.outputDeformationFieldFile.c_str() );
    fieldWriter->SetInput( defField );
    fieldWriter->SetUseCompression( true );

    try
      {
      fieldWriter->Update();
      }
    catch( itk::ExceptionObject& err )
      {
      std::cout << "Unexpected error." << std::endl;
      std::cout << err << std::endl;
      exit( EXIT_FAILURE );
      }
    }

  // write warped fiducials
  typedef typename DeformationFieldType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  string logfilename = args.outfidpath + PATH_SEP + "warpfid.log";
  ofstream logfile( logfilename.c_str() );

  std::vector< std::vector<float> > outfids( infids.size() );
  for(unsigned int i = 0; i < outfids.size(); ++i)
    {
    std::vector<float> f(3);
    f[0] = -infids[i][0];
    f[1] = -infids[i][1];
    f[2] = infids[i][2];

    PointType originalPoint;
    originalPoint[0] = f[0];
    originalPoint[1] = f[1];
    originalPoint[2] = f[2];

    IndexType index;
    invDefField->TransformPhysicalPointToIndex( originalPoint, index );
    VectorPixelType displacement = invDefField->GetPixel( index );
    f[0] += displacement[0];
    f[1] += displacement[1];
    f[2] += displacement[2];

    f[0] = -f[0];
    f[1] = -f[1];

    outfids[i] = f;
    logfile << index << endl;
    logfile << displacement << endl;
    }
  logfile.close();

  // save warped fiducials csv file
  string filename = args.outfidpath + PATH_SEP + args.outfidfile;
  ofstream outfile( filename.c_str() );
  if( outfile.fail() )
    {
    cout << "Error: could not open output for writing." << endl;
    exit( EXIT_FAILURE );
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

}

int main(int argc, char *argv[])
{
  // CLP magic
  PARSE_ARGS;

  arguments args;

  args.fixedImageFile = targetVolume;
  args.movingImageFile = sourceVolume;
  args.outputImageFile = outfidpath + PATH_SEP + "result.nrrd";
  args.outputDeformationFieldFile = outfidpath + PATH_SEP + "def.mha";
  args.numIterations.resize(3);
  args.numIterations[0] = 25;
  args.numIterations[1] = 10;
  args.numIterations[2] = 5;

  args.sigmaVel = 1.0;
  args.sigmaUp = 1.0;
  args.maxStepLength = 2.0;
  args.updateRule = 1;
  args.gradientType = 0;
  args.NumberOfBCHApproximationTerms = 2;
  args.useHistogramMatching = false;
  args.verbosity = 0;

  args.outfidpath = outfidpath;
  args.outfidfile = "output.fcsv";

  // read fidicials from a file
  std::ifstream ifs(infids_file.c_str());
  if (!ifs.is_open())
  {
	std::cout << "failed to open the fiducial file: " << infids_file << std::endl;
	return EXIT_FAILURE;
  }

  std::string fidid;
  float x, y, z;
  std::vector< float > xyz;
  std::vector< std::vector< float > > fids;
  while (true)
  {
	ifs >> fidid >> x >> y >> z;
	if (!ifs.good())
	    break;
	xyz.clear();
	xyz.push_back(x);
	xyz.push_back(y);
	xyz.push_back(z);
	fids.push_back(xyz);
  }

  //LogDomainDemonsRegistrationFunction<3>(args, infids);
  LogDomainDemonsRegistrationFunction<3>(args, fids);

  return EXIT_SUCCESS;
}

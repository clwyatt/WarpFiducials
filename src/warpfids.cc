#include <cstdlib>

#include <iostream>
using std::cout; using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <fstream>
using std::ifstream; using std::ofstream;

#include <sstream>
using std::istringstream;

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkPoint.h>
//#inlcude <itkWarpMeshFilter.h>

// See http://hdl.handle.net/10380/3060
#include <itkSymmetricLogDomainDemonsRegistrationFilter.h>
#include <itkMultiResolutionLogDomainDeformableRegistration.h>
#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>


/*
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkGridForwardWarpImageFilter.h>
#include <itkTransformToVelocityFieldSource.h>
#include <itkVectorCentralDifferenceImageFunction.h>
#include <itkWarpHarmonicEnergyCalculator.h>
#include <itkWarpImageFilter.h>
*/

struct Fiducial
{
  string name;
  float x, y, z;
  bool active, visible;
};

typedef vector<Fiducial> FiducialType;
typedef float PixelType;
typedef itk::Image<PixelType,3> ImageType;
typedef itk::Vector< PixelType, 3 > VectorPixelType;
typedef itk::Image< VectorPixelType, 3 > VelocityFieldType;
typedef itk::Image< VectorPixelType, 3 > DeformationFieldType;
typedef itk::Point< float, 3 > PointType;

bool readfid(const string & inputfidfile, FiducialType & infid);

int main(int argc, char *argv[])
{
  // command line options
  // warpfids inputfield.mha inputfids.fscv outputfids.fcsv
  if(argc != 5)
    {
    cout << "Error: missing option.\n" << endl;
    cout << "Usage: " << argv[0] << " target source inputfids.fscv outputfids.fcsv\n" << endl;
    return EXIT_FAILURE;
    }
  string targetfile = argv[1];
  string sourcefile = argv[2];
  string inputfidfile = argv[3];
  string outputfidfile = argv[4];

  // read source and target
  typedef itk::ImageFileReader< ImageType > ReaderType;
  ReaderType::Pointer source_reader = ReaderType::New();
  source_reader->SetFileName( sourcefile.c_str() );
  ReaderType::Pointer target_reader = ReaderType::New();
  target_reader->SetFileName( targetfile.c_str() );

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

  // read fiducials csv file
  FiducialType infid;
  if( readfid(inputfidfile, infid) )
    {
    cout << "Error reading input fiducial file. Halting." << endl;
    return EXIT_FAILURE;
    }

  // register source to target
  typedef itk::SymmetricLogDomainDemonsRegistrationFilter
    < ImageType, ImageType, VelocityFieldType>
    RegistrationFilterType;
  typedef RegistrationFilterType::GradientType GradientType;

  RegistrationFilterType::Pointer filter
    = RegistrationFilterType::New();

  filter->SetMaximumUpdateStepLength( 1.0 );
  filter->SetUseGradientType( static_cast<GradientType>(0) );
  filter->SetNumberOfBCHApproximationTerms( 2 );
  filter->SmoothVelocityFieldOn();
  filter->SetStandardDeviations( 1.5 );
  // filter->SmoothUpdateFieldOn();
  // filter->SetUpdateFieldStandardDeviations( 1.0 );

  typedef itk::MultiResolutionLogDomainDeformableRegistration
    < ImageType, ImageType, VelocityFieldType, PixelType > MultiResRegistrationFilterType;
  MultiResRegistrationFilterType::Pointer multires = MultiResRegistrationFilterType::New();

  typedef itk::VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<
     VelocityFieldType,double> FieldInterpolatorType;

  FieldInterpolatorType::Pointer VectorInterpolator = FieldInterpolatorType::New();

  multires->GetFieldExpander()->SetInterpolator(VectorInterpolator);

  multires->SetRegistrationFilter( filter );

  vector<unsigned int> numIterations(3);
  numIterations[0] = 15;
  numIterations[1] = 10;
  numIterations[2] = 5;

  multires->SetNumberOfLevels( numIterations.size() );

  multires->SetNumberOfIterations( &numIterations[0] );

  multires->SetFixedImage( target_reader->GetOutput() );
  multires->SetMovingImage( source_reader->GetOutput() );

  // Compute the deformation field
  try
    {
    multires->UpdateLargestPossibleRegion();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << "Unexpected error." << std::endl;
    std::cout << err << std::endl;
    return EXIT_FAILURE;
    }

  cout << "Registration Complete." << endl;

  // Final deformation field
  DeformationFieldType::Pointer defField = multires->GetDeformationField();
//  defField->DisconnectPipeline();

  // Write output deformation field
  typedef itk::ImageFileWriter< DeformationFieldType > FieldWriterType;
  FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
  fieldWriter->SetFileName( "temp.vtk" );
  fieldWriter->SetInput( defField );

  try
    {
    fieldWriter->Update();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << "Unexpected error." << std::endl;
    std::cout << err << std::endl;
    }

  // warp fiducials
  FiducialType outfid = infid;
  typedef DeformationFieldType::IndexType IndexType;

  for(unsigned int i = 0; i < outfid.size(); ++i)
    {
    Fiducial p = outfid[i];
    PointType originalPoint;
    originalPoint[0] = p.x;
    originalPoint[1] = p.y;
    originalPoint[2] = p.z;
    IndexType index;
    defField->TransformPhysicalPointToIndex( originalPoint, index );
    VectorPixelType displacement = defField->GetPixel( index );
    cout << index << endl;
    cout << displacement << endl;
    p.x = p.x + displacement[0];
    p.y = p.y + displacement[1];
    p.z = p.z + displacement[2];
    outfid[i] = p;
    }

  // save warped fiducials csv file
  ofstream outfidfile( outputfidfile.c_str() );
  if( outfidfile.fail() )
    {
    cout << "Error: could not open output for writing." << endl;
    return EXIT_FAILURE;
    }

  for(unsigned int i = 0; i < outfid.size(); ++i)
    {
    Fiducial p = outfid[i];
    string name = "warp-" + p.name;
    outfidfile << name.c_str() << ","
	   << p.x << "," << p.y << "," << p.z << ","
	   << p.active << "," << p.visible << endl;
    }
  outfidfile.close();

  return EXIT_SUCCESS;
}

bool readfid(const string & inputfidfile, FiducialType & infid)
{
  bool err = false;
  ifstream infidfile( inputfidfile.c_str() );

  if(infidfile.fail() )
    {
    cout << "Could not open file: " << inputfidfile.c_str() << endl;
    return true;
    }

  while(!infidfile.eof() && infidfile.good() )
    {
    string line;
    getline(infidfile, line);

    if( line.empty() || (line[0] == '#') ) continue;

    istringstream iss(line);
    Fiducial p;
    getline(iss, p.name, ',');

    string field;
    getline(iss, field, ',');
    istringstream xstream(field);
    if(!(xstream >> p.x) )
      {
      cout << "Warning: error reading x field on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream ystream(field);
    if(!(ystream >> p.y) )
      {
      cout << "Warning: error reading y field on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream zstream(field);
    if(!(zstream >> p.z) )
      {
      cout << "Warning: error reading z field on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream astream(field);
    if(!(astream >> p.active) )
      {
      cout << "Warning: error reading active flag on line:\n" << line << endl;
      err = true;
      continue;
      }

    getline(iss, field, ',');
    istringstream vstream(field);
    if(!(vstream >> p.visible) )
      {
      cout << "Warning: error reading visible flag on line:\n" << line << endl;
      err = true;
      continue;
      }

    infid.push_back(p);
    }
  infidfile.close();

  return err;
}

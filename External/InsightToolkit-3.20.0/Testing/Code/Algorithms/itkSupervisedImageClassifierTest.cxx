/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkSupervisedImageClassifierTest.cxx,v $
  Language:  C++
  Date:      $Date: 2007-08-20 12:47:12 $
  Version:   $Revision: 1.12 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#pragma warning ( disable : 4288 )
#endif
// Insight classes
#include "itkImage.h"
#include "itkVector.h"
#include "vnl/vnl_matrix_fixed.h"
#include "itkImageRegionIterator.h"
#include "itkLightProcessObject.h"

#include "itkImageGaussianModelEstimator.h"
#include "itkMahalanobisDistanceMembershipFunction.h"
#include "itkMinimumDecisionRule.h"
#include "itkImageClassifierBase.h"


//Data definitons 
#define   IMGWIDTH            2
#define   IMGHEIGHT           2
#define   NFRAMES             4
#define   NUMBANDS            2  
#define   NDIMENSION          3
#define   NUM_CLASSES         3
#define   MAX_NUM_ITER       50


// class to support progress feeback
class ShowProgressObject
{
public:
  ShowProgressObject(itk::LightProcessObject * o)
    {m_Process = o;}
  void ShowProgress()
    {std::cout << "Progress " << m_Process->GetProgress() << std::endl;}
  itk::LightProcessObject::Pointer m_Process;
};



int itkSupervisedImageClassifierTest(int, char* [] )
{

  //------------------------------------------------------
  //Create a simple test image with width, height, and 
  //depth 4 vectors each with each vector having data for 
  //2 bands.
  //------------------------------------------------------
  typedef itk::Image<itk::Vector<double,NUMBANDS>,NDIMENSION> VecImageType; 

  VecImageType::Pointer vecImage = VecImageType::New();

  VecImageType::SizeType vecImgSize = {{ IMGWIDTH , IMGHEIGHT, NFRAMES }};

  VecImageType::IndexType index;
  index.Fill(0);
  VecImageType::RegionType region;

  region.SetSize( vecImgSize );
  region.SetIndex( index );

  vecImage->SetLargestPossibleRegion( region );
  vecImage->SetBufferedRegion( region );
  vecImage->Allocate();

  // setup the iterators
  typedef VecImageType::PixelType VecImagePixelType;

  enum { VecImageDimension = VecImageType::ImageDimension };
  typedef
    itk::ImageRegionIterator< VecImageType > VecIterator;

  VecIterator outIt( vecImage, vecImage->GetBufferedRegion() );

  //--------------------------------------------------------------------------
  //Manually create and store each vector
  //--------------------------------------------------------------------------

  //Slice 1
  //Vector no. 1
  VecImagePixelType vec;
  vec.Fill(21); outIt.Set( vec ); ++outIt;
  //Vector no. 2
  vec.Fill(20); outIt.Set( vec ); ++outIt;
  //Vector no. 3
  vec.Fill(8); outIt.Set( vec ); ++outIt;
  //Vector no. 4
  vec.Fill(10); outIt.Set( vec ); ++outIt;
  //Slice 2
  //Vector no. 1
  vec.Fill(22); outIt.Set( vec ); ++outIt;
  //Vector no. 2
  vec.Fill(21); outIt.Set( vec ); ++outIt;
  //Vector no. 3
  vec.Fill(11); outIt.Set( vec ); ++outIt;
  //Vector no. 4
  vec.Fill(9); outIt.Set( vec ); ++outIt;
  
  //Slice 3
  //Vector no. 1 
  vec.Fill(19); outIt.Set( vec ); ++outIt;
  //Vector no. 2
  vec.Fill(19); outIt.Set( vec ); ++outIt;
  //Vector no. 3
  vec.Fill(11); outIt.Set( vec ); ++outIt;
  //Vector no. 4
  vec.Fill(11); outIt.Set( vec ); ++outIt;
  
  //Slice 4
  //Vector no. 1
  vec.Fill(18); outIt.Set( vec ); ++outIt;
  //Vector no. 2
  vec.Fill(18); outIt.Set( vec ); ++outIt;
  //Vector no. 3
  vec.Fill(12); outIt.Set( vec ); ++outIt;
  //Vector no. 4
  vec.Fill(14); outIt.Set( vec ); ++outIt;

  //---------------------------------------------------------------
  //Generate the training data
  //---------------------------------------------------------------
  typedef itk::Image<unsigned short,NDIMENSION> ClassImageType; 
  ClassImageType::Pointer classImage  = ClassImageType::New();

  ClassImageType::SizeType classImgSize = {{ IMGWIDTH , IMGHEIGHT, NFRAMES }};

  ClassImageType::IndexType classindex;
  classindex.Fill(0);

  ClassImageType::RegionType classregion;

  classregion.SetSize( classImgSize );
  classregion.SetIndex( classindex );

  classImage->SetLargestPossibleRegion( classregion );
  classImage->SetBufferedRegion( classregion );
  classImage->Allocate();

  // setup the iterators
  typedef ClassImageType::PixelType ClassImagePixelType;

  typedef
    itk::ImageRegionIterator<ClassImageType> ClassImageIterator;

  ClassImageIterator 
    classoutIt( classImage, classImage->GetBufferedRegion() );



  ClassImagePixelType outputPixel;
  //--------------------------------------------------------------------------
  //Manually create and store each vector
  //--------------------------------------------------------------------------
  //Slice 1
  //Pixel no. 1
  outputPixel = 2;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 2 
  outputPixel = 2;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 3
  outputPixel = 1;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 4
  outputPixel = 1;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Slice 2
  //Pixel no. 1
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;
  
  //Pixel no. 2
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 3
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 4
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Slice 3
  //Pixel no. 1 
  outputPixel = 2;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 2
  outputPixel = 2;
  classoutIt.Set( outputPixel );
  ++classoutIt;
  
  //Pixel no. 3
  outputPixel = 1;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 4
  outputPixel = 1;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Slice 4
  //Pixel no. 1
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;
  
  //Pixel no. 2
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 3
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //Pixel no. 4
  outputPixel = 0;
  classoutIt.Set( outputPixel );
  ++classoutIt;

  //----------------------------------------------------------------------
  //Set membership function (Using the statistics objects)
  //----------------------------------------------------------------------
  namespace stat = itk::Statistics;

  typedef stat::MahalanobisDistanceMembershipFunction< VecImagePixelType > 
    MembershipFunctionType ;
  typedef MembershipFunctionType::Pointer MembershipFunctionPointer ;

  typedef std::vector< MembershipFunctionPointer > 
    MembershipFunctionPointerVector;

  //----------------------------------------------------------------------
  //Set the image model estimator
  //----------------------------------------------------------------------
  typedef itk::ImageGaussianModelEstimator<VecImageType,
    MembershipFunctionType, ClassImageType> 
    ImageGaussianModelEstimatorType;
  
  ImageGaussianModelEstimatorType::Pointer 
    applyEstimateModel = ImageGaussianModelEstimatorType::New();  

  applyEstimateModel->SetNumberOfModels(NUM_CLASSES);
  applyEstimateModel->SetInputImage(vecImage);
  applyEstimateModel->SetTrainingImage(classImage);  

  //Run the gaussian classifier algorithm
  applyEstimateModel->Update();
  applyEstimateModel->Print(std::cout); 

  MembershipFunctionPointerVector membershipFunctions = 
    applyEstimateModel->GetMembershipFunctions();  

  for(unsigned int idx=0; idx < membershipFunctions.size(); idx++ )
    {
    std::cout << "Number of samples for class " << idx << " is " <<
      membershipFunctions[ idx ]->GetNumberOfSamples() << std::endl;
    }

  //----------------------------------------------------------------------
  //Set the decision rule 
  //----------------------------------------------------------------------  
  typedef itk::DecisionRuleBase::Pointer DecisionRuleBasePointer;

  typedef itk::MinimumDecisionRule DecisionRuleType;
  DecisionRuleType::Pointer  
    myDecisionRule = DecisionRuleType::New();

  //----------------------------------------------------------------------
  // Test code for the supervised classifier algorithm
  //----------------------------------------------------------------------

  //---------------------------------------------------------------------
  // Multiband data is now available in the right format
  //---------------------------------------------------------------------
  typedef VecImagePixelType MeasurementVectorType;

  typedef itk::ImageClassifierBase< VecImageType,
    ClassImageType > SupervisedClassifierType;

  SupervisedClassifierType::Pointer 
    applyClassifier = SupervisedClassifierType::New();
 
  typedef ShowProgressObject 
    ProgressType;

  ProgressType progressWatch(applyClassifier);
  itk::SimpleMemberCommand<ProgressType>::Pointer command;
  command = itk::SimpleMemberCommand<ProgressType>::New();
  command->SetCallbackFunction(&progressWatch,
                               &ProgressType::ShowProgress);
  applyClassifier->AddObserver(itk::ProgressEvent(), command);

  // Set the Classifier parameters
  applyClassifier->SetNumberOfClasses(NUM_CLASSES);
  applyClassifier->SetInputImage(vecImage);

  // Set the decison rule 
  applyClassifier->
    SetDecisionRule((DecisionRuleBasePointer) myDecisionRule );

  //Add the membership functions
  for( unsigned int i=0; i<NUM_CLASSES; i++ )
    {
    applyClassifier->AddMembershipFunction( membershipFunctions[i] );
    }

  //Run the gaussian classifier algorithm
  applyClassifier->Update();

  //Get the classified image
  ClassImageType::Pointer 
    outClassImage = applyClassifier->GetClassifiedImage();

  applyClassifier->Print(std::cout); 

  //Print the gaussian classified image
  ClassImageIterator labeloutIt( outClassImage, 
                                 outClassImage->GetBufferedRegion() );

  int i=0;
  while(!labeloutIt.IsAtEnd())
    {
    //Print the classified index
    int classIndex = (int) labeloutIt.Get();
    std::cout << " Pixel No " << i << " Value " << classIndex << std::endl;
    ++i;
    ++labeloutIt;
    }//end while

  //Verify if the results were as per expectation
  labeloutIt.GoToBegin();
  bool passTest = true;

  //Loop through the data set
  while(!labeloutIt.IsAtEnd())
    {
    int classIndex = (int) labeloutIt.Get();
    if (classIndex != 2)
      {
      passTest = false;
      break;
      }
    ++labeloutIt;

    classIndex = (int) labeloutIt.Get();
    if (classIndex != 2)
      {
      passTest = false;
      break;
      }
    ++labeloutIt;

    classIndex = (int) labeloutIt.Get();
    if (classIndex != 1)
      {
      passTest = false;
      break;
      }
    ++labeloutIt;

    classIndex = (int) labeloutIt.Get();
    if (classIndex != 1)
      {
      passTest = false;
      break;
      }
    ++labeloutIt;

    }//end while

  if( passTest == true )
    {
    std::cout<< "Supervised Classifier Test Passed" << std::endl;
    }
  else
    {
    std::cout<< "Supervised Classifier Test failed" << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

/**
* @file   Multiresolution-SRS3D.cxx
* @author Tobias Gass <tobiasgass@gmail.com>
* @date   Wed Mar 11 14:55:33 2015
* 
* @brief  Example for 3D SRS using handcrafted segmentation 
* 
* 
*/

#include <stdio.h>
#include <iostream>

#include "ArgumentParser.h"

#include "SRSConfig.h"
#include "HierarchicalSRSImageToImageFilter.h"
#include "Graph.h"
#include "FastGraph.h"
#include "BaseLabel.h"
#include "Metrics.h"
#include "Potential-Registration-Unary.h"
#include "Potential-Registration-Pairwise.h"
#include "Potential-Segmentation-Unary.h"
#include "Potential-Coherence-Pairwise.h"
#include "Potential-Segmentation-Pairwise.h"
#include "Log.h"
#include "Preprocessing.h"
#include "TransformationUtils.h"



using namespace std;
using namespace SRS;
using namespace itk;


bool (*realProgressCallbackFunc)(int progress);
bool MyProgressCallback(int progress) {
  return realProgressCallbackFunc(progress);
}


//////////////////////////////////////////////////
// The main registration algorithm entry function
//////////////////////////////////////////////////
EXTERN_C __declspec(dllexport) wchar_t* __cdecl DoRegistration(
  int             sourceSizeX,
  int             sourceSizeY,   
  int             sourceSizeZ,
  double          sourceResX,
  double          sourceResY,
  double          sourceResZ,
  unsigned short* sourcePixels,
  int             sourceModality,
  double          sourceSlope,
  double          sourceIntercept,
  int             targetSizeX,
  int             targetSizeY,
  int             targetSizeZ,
  double          targetResX,
  double          targetResY,
  double          targetResZ,
  unsigned short* targetPixels,
  int             targetModality,
  double          targetSlope,
  double          targetIntercept,
  double*         rigidAlignment,
  double          voiOriginX,
  double          voiOriginY,
  double          voiOriginZ,
  double          voiExtentX,
  double          voiExtentY,
  double          voiExtentZ,
  int             numOptionalArguments,
  wchar_t**       optionalArguments,
  int*            fieldSizeX,
  int*            fieldSizeY,
  int*            fieldSizeZ,
  double*         fieldResX,
  double*         fieldResY,
  double*         fieldResZ,
  double*         fieldOriginX,
  double*         fieldOriginY,
  double*         fieldOriginZ,
  float**         fieldVectors,
  bool            (*progressCallbackFunc)(int progress)
  )
{
  const bool DEFORM_ATLAS_IMAGE = false;
  const bool WRITE_IMAGES_OUT = false;
  // re-route the progress callback
  realProgressCallbackFunc = progressCallbackFunc;

  int progress = 0;
  realProgressCallbackFunc(progress);

  //get config file for SRS params
  SRSConfig::Pointer filterConfig = SRSConfig::New();
  filterConfig->parseFile("C:\\_SmartSeg\\srsConfigs.txt");

  progress = 3;
  realProgressCallbackFunc(progress);

  //read optional arguments
  wstring firstString(optionalArguments[1]);
  string atlasDir(firstString.begin(), firstString.end());

  wstring secondString(optionalArguments[3]);
  string targetDir(secondString.begin(), secondString.end());

  wstring thirdString(optionalArguments[5]);
  string protocolDir(thirdString.begin(), thirdString.end());

  //re-route standard output to protocol file
  std::string atlasName = atlasDir.substr(atlasDir.length()-9,9); 
  std::string tagetName = targetDir.substr(targetDir.length()-9,9); 
  std::string protocolFile = protocolDir + "\\SRS_" + atlasName + "_" + tagetName + ".txt";
  std::string landmarkFile = protocolDir + "\\landmarkDist_" + atlasName + "_" + tagetName + ".txt";
  std::ofstream out(protocolFile);
  std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
  std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!

  filterConfig->atlasLandmarkFilename = atlasDir + "\\landmarks.txt";
  filterConfig->targetLandmarkFilename = targetDir + "\\landmarks.txt";

  LOG<< " target landmark filename: " << filterConfig->targetLandmarkFilename << std::endl;
  
  //set the correct mask filename (but defined in the SRS Config file to see afterwards if a mask was used or not)
  if (filterConfig->atlasMaskFilename!="") {
    filterConfig->atlasMaskFilename = atlasDir + "\\" + filterConfig->atlasMaskFilename;
    LOG<<"Mask Image: " + filterConfig->atlasMaskFilename <<std::endl;
  } else {
    LOG<<"NO MASK IMAGE"<<std::endl;
  }


  if (filterConfig->logFileName!=""){
    mylog.setCachedLogging();
  }

  progress = 5;
  realProgressCallbackFunc(progress);

  logSetStage("Init");

  progress = 6;
  realProgressCallbackFunc(progress);

  typedef float PixelType;
  const unsigned int D=3;
  typedef Image<PixelType,D> ImageType;
  typedef ImageType::Pointer ImagePointerType;
  typedef Image<float, D> FloatImageType;
  typedef ImageType::ConstPointer ImageConstPointerType;
  typedef TransfUtils<ImageType>::DisplacementType DisplacementType;
  typedef SparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
  //typedef SemiSparseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
  //typedef DenseRegistrationLabelMapper<ImageType,DisplacementType> LabelMapperType;
  typedef TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
  typedef DeformationFieldType::Pointer DeformationFieldPointerType;




  typedef UnaryPotentialSegmentationBoneMarcel< ImageType > SegmentationUnaryPotentialType;

  typedef PairwisePotentialSegmentationMarcel<ImageType> SegmentationPairwisePotentialType;

  // //reg
  typedef MultiThreadedLocalSimilarityNCC<FloatImageType, ImageType> SimilarityType;
  //typedef MultiThreadedLocalSimilaritySSD<FloatImageType, ImageType> SimilarityType;
  typedef UnaryRegistrationPotentialWithCaching< ImageType, SimilarityType > RegistrationUnaryPotentialType;

  typedef PairwisePotentialRegistration< ImageType > RegistrationPairwisePotentialType;

  typedef PairwisePotentialCoherence< ImageType > CoherencePairwisePotentialType;
  //typedef PairwisePotentialMultilabelCoherence< ImageType > CoherencePairwisePotentialType;



  typedef FastGraphModel<ImageType,RegistrationUnaryPotentialType,RegistrationPairwisePotentialType,SegmentationUnaryPotentialType,SegmentationPairwisePotentialType,CoherencePairwisePotentialType>        GraphType;


  typedef HierarchicalSRSImageToImageFilter<GraphType>        FilterType;    
  //create filter
  FilterType::Pointer filter=FilterType::New();
  filter->setConfig(filterConfig);

  logSetStage("Instantiate Potentials");


  RegistrationUnaryPotentialType::Pointer unaryRegistrationPot=RegistrationUnaryPotentialType::New();
  SegmentationUnaryPotentialType::Pointer unarySegmentationPot=SegmentationUnaryPotentialType::New();
  RegistrationPairwisePotentialType::Pointer pairwiseRegistrationPot=RegistrationPairwisePotentialType::New();
  SegmentationPairwisePotentialType::Pointer pairwiseSegmentationPot=SegmentationPairwisePotentialType::New();
  CoherencePairwisePotentialType::Pointer pairwiseCoherencePot=CoherencePairwisePotentialType::New();

  filter->setUnaryRegistrationPotentialFunction((unaryRegistrationPot));
  filter->setPairwiseRegistrationPotentialFunction((pairwiseRegistrationPot));
  filter->setUnarySegmentationPotentialFunction((unarySegmentationPot));
  filter->setPairwiseCoherencePotentialFunction((pairwiseCoherencePot));
  filter->setPairwiseSegmentationPotentialFunction((pairwiseSegmentationPot));

  filter->setLandmarkFilename(landmarkFile);

  logUpdateStage("IO");
  logSetVerbosity(filterConfig->verbose);
  //************************************************************************************************************//
  //******************************Target image *****************************************************************//
  //************************************************************************************************************//
  ImageType::Pointer targetImage = ImageType::New();

  ImageType::SizeType  targetSize;
  targetSize[0]  = targetSizeX;  // size along X
  targetSize[1]  = targetSizeY;  // size along Y
  targetSize[2]  = targetSizeZ;  // size along Z

  LOGV(6) << "target size x: " << targetSize[0] <<std::endl;
  LOGV(6) << "target size y: " << targetSize[1] <<std::endl;
  LOGV(6) << "target size z: " << targetSize[2] <<std::endl;

  ImageType::IndexType start;
  start[0] =   0;  // first index on X
  start[1] =   0;  // first index on Y
  start[2] =   0;  // first index on Z

  ImageType::RegionType targetRegion;
  targetRegion.SetSize( targetSize );
  targetRegion.SetIndex( start );

  targetImage->SetRegions(targetRegion);
  targetImage->Allocate();

  double targetSpacing[3];
  targetSpacing[0] = targetResX;
  targetSpacing[1] = targetResY;
  targetSpacing[2] = targetResZ;
  targetImage ->SetSpacing(targetSpacing);

  LOGV(6) << "target res x: " << targetSpacing[0] <<std::endl;
  LOGV(6) << "target res y: " << targetSpacing[1] <<std::endl;
  LOGV(6) << "target res z: " << targetSpacing[2] <<std::endl;

  double targetOrigin[3];
  for (int d = 0; d<3; ++d){
    targetOrigin[d] = -0.5*targetSpacing[d] * targetSize[d];
  }
  targetImage ->SetOrigin(targetOrigin);

  itk::ImageRegionIteratorWithIndex<ImageType> targetIterator(targetImage,targetImage->GetLargestPossibleRegion());
  int offset = 0;
  for (targetIterator.GoToBegin();!targetIterator.IsAtEnd();++targetIterator){
    targetIterator.Set(targetPixels[offset++]);
  }

  LOGI(6,ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_targetImageBeforeCut.nii", targetImage));

  //set original extent to know afterwards where the landmarks are
  std::vector<double> originalTargetExtent(D);
  ImageType::SpacingType originalSpacing = targetImage->GetSpacing();
  ImageType::SizeType originalSize = targetImage->GetLargestPossibleRegion().GetSize();
  for (int d=0;d<D;++d) {
    originalTargetExtent[d] = originalSpacing[d] * originalSize[d];
  }


  //************************************************************************************************************//
  //******************************VOI cropping *****************************************************************//
  //************************************************************************************************************//
#define USE_VOI
#ifdef USE_VOI
  //crop image to ROI
  itk::ResampleImageFilter<ImageType, ImageType>::Pointer resampleFilter = itk::ResampleImageFilter<ImageType, ImageType>::New();
  typedef itk::ResampleImageFilter<ImageType, ImageType>::PointType PointType;
  resampleFilter->SetInput(targetImage);
  //convert MIRS origin to ITK origin
  //MIRS origin is relative position compared to the 'center' of the region of the target image, ITK is 'top left' corner
  PointType voiExtent; voiExtent[0] = voiExtentX; voiExtent[1] = voiExtentY; voiExtent[2] = voiExtentZ;
  PointType voiOrigin; voiOrigin[0] = voiOriginX; voiOrigin[1] = voiOriginY; voiOrigin[2] = voiOriginZ;
  for (int d = 0; d<3; ++d)
  {
    voiOrigin[d] = voiOrigin[d] - 0.5*voiExtent[d]; 
  }
       
  LOGV(1) << VAR(voiOriginX) << " " << VAR(voiOriginY) << " " << VAR(voiOriginZ) << std::endl; 
  LOGV(1) << VAR(voiExtentX) << " " << VAR(voiExtentY) << " " << VAR(voiExtentZ) << std::endl;
  ImageType::SizeType voiSize;

  for (int d = 0; d<3; ++d) { voiSize[d] = voiExtent[d] / targetSpacing[d]; }
  LOG << "voiOrigin " << voiOrigin << " voiExtent: " << voiExtent << " " << voiExtentX << std::endl;
  resampleFilter->SetOutputOrigin(voiOrigin);
  resampleFilter->SetOutputDirection(targetImage->GetDirection());
  resampleFilter->SetOutputSpacing(targetImage->GetSpacing());

  resampleFilter->SetSize(voiSize);
  resampleFilter->Update();
  targetImage = resampleFilter->GetOutput();
#endif

  LOGI(6, ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_targetImageAfterCut.nii", targetImage));


  //************************************************************************************************************//
  //******************************Resampling of target image****************************************************//
  //************************************************************************************************************//
  
  //needs to duplicate original image before resampling, since the deformation field is interpolated this resolution
  ImagePointerType originalTargetImage = ImageUtils<ImageType>::duplicate( targetImage);

  //Resample target Image
  LOGI(6,ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_VOIbeforeResampling.nii", targetImage));
  targetImage = FilterUtils<ImageType>::ConstantResample(targetImage);
  LOGI(6,ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_VOIafterResampling.nii", targetImage));

  if (filterConfig->verbose>5) {
    ImageType::SpacingType newTargetSpacing = targetImage->GetSpacing();
    LOG << "target spacing after resample x: " << newTargetSpacing[0] <<std::endl;
    LOG << "target spacing after resample y: " << newTargetSpacing[1] <<std::endl;
    LOG << "target spacing after resample z: " << newTargetSpacing[2] <<std::endl;

    ImageType::SizeType newTargetSize = targetImage->GetLargestPossibleRegion().GetSize();
    LOG << "target size after resample x: " << newTargetSize[0] <<std::endl;
    LOG << "target size after resample y: " << newTargetSize[1] <<std::endl;
    LOG << "target size after resample z: " << newTargetSize[2] <<std::endl;
  }

  if (!targetImage) {LOG<<"failed!"<<endl; exit(0);}

#if 0
  if (filterConfig->normalizeImages){
    targetImage=FilterUtils<ImageType>::normalizeImage(targetImage);
  }
#endif
  //************************************************************************************************************//
  //******************************Atlas image *****************************************************************//
  //************************************************************************************************************//

  ImagePointerType atlasImage = ImageType::New();

  ImageType::SizeType  size;
  size[0]  = sourceSizeX;  // size along X
  size[1]  = sourceSizeY;  // size along Y
  size[2]  = sourceSizeZ;  // size along Z

  LOGV(6) << "source size x: " << size[0] <<std::endl;
  LOGV(6) << "source size y: " << size[1] <<std::endl;
  LOGV(6) << "source size z: " << size[2] <<std::endl;

  ImageType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );

  atlasImage->SetRegions( region );
  atlasImage->Allocate();

  double spacing[3];
  spacing[0] = sourceResX;
  spacing[1] = sourceResY;
  spacing[2] = sourceResZ;

  LOGV(6) << "source res x: " << spacing[0] <<std::endl;
  LOGV(6) << "source res y: " << spacing[1] <<std::endl;
  LOGV(6) << "source res z: " << spacing[2] <<std::endl;

  atlasImage ->SetSpacing(spacing);

  double sourceOrigin[3];
  for (int d = 0; d<3; ++d){
    sourceOrigin[d] = -0.5*spacing[d] * size[d];
  }
  atlasImage->SetOrigin(sourceOrigin);
  LOGV(6) << "source origin: " << sourceOrigin[0] << " " << sourceOrigin[1] << " " << sourceOrigin[2] <<std::endl;


  itk::ImageRegionIteratorWithIndex<ImageType> sourceIterator(atlasImage,atlasImage->GetLargestPossibleRegion());
  offset = 0;
  for (sourceIterator.GoToBegin();!sourceIterator.IsAtEnd();++sourceIterator){
    sourceIterator.Set(sourcePixels[offset++]);
  }

  ImagePointerType originalAtlasImage= ImageUtils<ImageType>::duplicate(atlasImage);


  //************************************************************************************************************//
  //******************************Rigid transformation *********************************************************//
  //************************************************************************************************************//

  typedef itk::AffineTransform<double,3> AffineTransformType;
  AffineTransformType::Pointer affine = AffineTransformType::New();
  AffineTransformType::MatrixType matrix;
  AffineTransformType::OutputVectorType translation;

  matrix(0, 0) = rigidAlignment[0];
  matrix(0, 1) = rigidAlignment[1];
  matrix(0, 2) = rigidAlignment[2];
  translation[0] = rigidAlignment[3];
  matrix(1, 0) = rigidAlignment[4];
  matrix(1, 1) = rigidAlignment[5];
  matrix(1, 2) = rigidAlignment[6];
  translation[1] = rigidAlignment[7];
  matrix(2, 0) = rigidAlignment[8];
  matrix(2, 1) = rigidAlignment[9];
  matrix(2, 2) = rigidAlignment[10];
  translation[2] = rigidAlignment[11];

  affine->SetTranslation(translation);
  affine->SetMatrix(matrix);

  LOGV(1) << "affine transformation:" << std::endl;
  LOGV(1) << VAR(affine) << std::endl;
  DeformationFieldPointerType transf = TransfUtils<ImageType>::affineToDisplacementField(affine, targetImage);
  filter->setBulkTransform(transf);

  if (filterConfig->verbose>5) {
      ImagePointerType deformedAtlasAffine = TransfUtils<ImageType>::affineDeformImage(atlasImage, affine, targetImage);
      ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_atlasDeformedRigid.nii", deformedAtlasAffine);
      ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_atlasUnDeformedRigid.nii", atlasImage);
  }

  if (!atlasImage) {
    LOG<<"Warning: no atlas image loaded!"<<endl;
    LOG<<"Loading atlas segmentation image :"<<filterConfig->atlasSegmentationFilename<<std::endl;
  }
  ImagePointerType atlasSegmentation;
  if (filterConfig->atlasSegmentationFilename !="")atlasSegmentation=ImageUtils<ImageType>::readImage(filterConfig->atlasSegmentationFilename);
  if (!atlasSegmentation) {LOG<<"Warning: no atlas segmentation loaded!"<<endl; }

  ImagePointerType targetAnatomyPrior;
  if (filterConfig->targetAnatomyPriorFilename !="") {
    targetAnatomyPrior=ImageUtils<ImageType>::readImage(filterConfig->targetAnatomyPriorFilename);
    filterConfig->useTargetAnatomyPrior=true;
  }

  ImagePointerType atlasMaskImage=NULL;
  if (filterConfig->atlasMaskFilename!="") {
    atlasMaskImage=ImageUtils<ImageType>::readImage(filterConfig->atlasMaskFilename);
    LOGI(6, ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_maskBeforeTransform.nii",atlasMaskImage));
    atlasMaskImage->SetOrigin(atlasImage->GetOrigin());
    LOGI(6, ImageUtils<ImageType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_maskAfterTransform.nii",atlasMaskImage));
  }


  logResetStage;
  logSetStage("Preprocessing");

  if (filterConfig->histNorm){
    // Histogram match the images
    typedef itk::HistogramMatchingImageFilter<ImageType,ImageType> HEFilterType;
    HEFilterType::Pointer IntensityEqualizeFilter = HEFilterType::New();
    IntensityEqualizeFilter->SetReferenceImage(targetImage  );
    IntensityEqualizeFilter->SetInput( atlasImage );
    IntensityEqualizeFilter->SetNumberOfHistogramLevels( 100);
    IntensityEqualizeFilter->SetNumberOfMatchPoints( 15);
    IntensityEqualizeFilter->ThresholdAtMeanIntensityOn();
    IntensityEqualizeFilter->Update();
    atlasImage=IntensityEqualizeFilter->GetOutput();
  }

  //preprocessing 1: gradients
  ImagePointerType targetGradient, atlasGradient;
  if (filterConfig->segment){
    if (filterConfig->targetGradientFilename!=""){
      targetGradient=(ImageUtils<ImageType>::readImage(filterConfig->targetGradientFilename));
    }else{
      targetGradient=Preprocessing<ImageType>::computeSheetness(targetImage);
      if (WRITE_IMAGES_OUT) {
        LOGI(8,ImageUtils<ImageType>::writeImage("targetsheetness.nii",targetGradient));  
      }
    }
    if (filterConfig->atlasGradientFilename!=""){
      atlasGradient=(ImageUtils<ImageType>::readImage(filterConfig->atlasGradientFilename));
    }else{
      if (atlasImage.IsNotNull()){
        atlasGradient=Preprocessing<ImageType>::computeSheetness(atlasImage);
        if (WRITE_IMAGES_OUT) {
          LOGI(8,ImageUtils<ImageType>::writeImage("atlassheetness.nii",atlasGradient));
        }
      }
    }

    if (filterConfig->useTargetAnatomyPrior && ! targetAnatomyPrior.IsNotNull() ){
      //targetAnatomyPrior=Preprocessing<ImageType>::computeSoftTargetAnatomyEstimate(targetImage);
      LOG<<"NOT YET IMPLEMENTED: Preprocessing<ImageType>::computeSoftTargetAnatomyEstimate"<<endl;
      exit(0);
    }
    //preprocessing 2: multilabel
    if (filterConfig->computeMultilabelAtlasSegmentation){
      atlasSegmentation=FilterUtils<ImageType>::computeMultilabelSegmentation(atlasSegmentation);
      filterConfig->nSegmentations=5;//TODO!!!!
    }
  }
  logResetStage;


  ImagePointerType originalAtlasSegmentation=atlasSegmentation;
  //preprocessing 3: downscaling

  if (filterConfig->downScale<1){
    double scale=filterConfig->downScale;
    LOG<<"Resampling images from "<< targetImage->GetLargestPossibleRegion().GetSize()<<" by a factor of"<<scale<<endl;
    targetImage=FilterUtils<ImageType>::LinearResample(targetImage,scale,true);
    if (atlasImage.IsNotNull()) atlasImage=FilterUtils<ImageType>::LinearResample(atlasImage,scale,true);
    if (atlasMaskImage.IsNotNull()) atlasMaskImage=FilterUtils<ImageType>::NNResample(atlasMaskImage,scale,false);
    if (atlasSegmentation.IsNotNull()) {
      atlasSegmentation=FilterUtils<ImageType>::NNResample((atlasSegmentation),scale,false);
      //ImageUtils<ImageType>::writeImage("testA.nii",atlasSegmentation);
    }
    if (filterConfig->segment){
      LOGV(3)<<"Resampling gradient images and anatomy prior by factor of "<<scale<<endl;
      targetGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)targetGradient),scale,true);
      atlasGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)atlasGradient),scale,true);
      //targetGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetGradient,sigma),scale);
      //atlasGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)atlasGradient,sigma),scale);
      if (filterConfig->useTargetAnatomyPrior){
        targetAnatomyPrior=FilterUtils<ImageType>::NNResample((targetAnatomyPrior),scale,false);
      }

    }
  }
  if (filterConfig->segment){
    if (atlasGradient.IsNotNull()) { 
      LOGI(10,ImageUtils<ImageType>::writeImage("atlassheetness.nii",atlasGradient));
    }
    LOGI(10,ImageUtils<ImageType>::writeImage("targetsheetness.nii",targetGradient));
  }


  logResetStage;
  filter->setTargetImage(targetImage);
  filter->setTargetGradient(targetGradient);
  filter->setAtlasImage(atlasImage);
  filter->setAtlasMaskImage(atlasMaskImage);
  filter->setAtlasGradient(atlasGradient);
  filter->setAtlasSegmentation(atlasSegmentation);
  if (filterConfig->useTargetAnatomyPrior){
    filter->setTargetAnatomyPrior(targetAnatomyPrior);
  }
  filter->setOriginalTargetExtent(originalTargetExtent);
  logSetStage("Bulk transforms");

  if (filterConfig->affineBulkTransform!=""){
    TransfUtils<ImageType>::AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(filterConfig->affineBulkTransform);
    if (WRITE_IMAGES_OUT) {
      LOGI(8,ImageUtils<ImageType>::writeImage("def.nii",TransfUtils<ImageType>::affineDeformImage(originalAtlasImage,affine,originalTargetImage)));
    }
    //DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,originalTargetImage);
    DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,targetImage);
    if (WRITE_IMAGES_OUT) {
      LOGI(8,ImageUtils<ImageType>::writeImage("def2.nii",TransfUtils<ImageType>::warpImage((ImageType::ConstPointer)originalAtlasImage,transf)));
    }
    filter->setBulkTransform(transf);
  }
  else if (filterConfig->bulkTransformationField!=""){
    filter->setBulkTransform(ImageUtils<DeformationFieldType>::readImage(filterConfig->bulkTransformationField));
  }else if (filterConfig->initWithMoments){
    //LOG<<" NOT NOT NOT Computing transform to move image centers on top of each other.."<<std::endl;
    LOG<<"initializing deformation using moments.."<<std::endl;
    DeformationFieldPointerType transf=TransfUtils<ImageType>::computeCenteringTransform(originalTargetImage,originalAtlasImage);
    filter->setBulkTransform(transf);
  }
  logResetStage;//bulk transforms
  //originalTargetImage=NULL;
  //originalAtlasImage=NULL;
  // compute SRS
  clock_t FULLstart = clock();
  filter->Init();
  logResetStage; //init

  filter->Update();
  LOG << "after update" << endl;
  logSetStage("Finalizing");
  clock_t FULLend = clock();
  float t = (float) ((double)(FULLend - FULLstart) / CLOCKS_PER_SEC);
  LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
  LOG<<"Unaries: "<<tUnary<<" Optimization: "<<tOpt<<std::endl;	
  LOG<<"Pairwise: "<<tPairwise<<std::endl;


  //process outputs
  ImagePointerType targetSegmentationEstimate=filter->getTargetSegmentationEstimate();
  DeformationFieldPointerType finalDeformation=filter->getFinalDeformation();

  //delete filter;
  //if (filterConfig.atlasFilename!="") originalAtlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
  //if (filterConfig.targetFilename!="") originalTargetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);

  //upsample?
  if (filterConfig->downScale<1){
    LOG<<"Upsampling Images.."<<endl;

    //it would probably be far better to create a surface for each label, 'upsample' that surface, and then create a binary volume for each surface which are merged in a last step
    if (targetSegmentationEstimate){
#if 0            
      targetSegmentationEstimate=FilterUtils<ImageType>::NNResample(targetSegmentationEstimate,originalTargetImage,false);
#else
      targetSegmentationEstimate=FilterUtils<ImageType>::upsampleSegmentation(targetSegmentationEstimate,originalTargetImage);
#endif
    }
  }

  if (targetSegmentationEstimate.IsNotNull()){
    if (WRITE_IMAGES_OUT) {
      ImageUtils<ImageType>::writeImage(filterConfig->segmentationOutputFilename,targetSegmentationEstimate);
    }
  }

  if (finalDeformation.IsNotNull() ) {
    if (filterConfig->defFilename!="")
      ImageUtils<DeformationFieldType>::writeImage(filterConfig->defFilename,finalDeformation);
    if (filterConfig->linearDeformationInterpolation){
      finalDeformation=TransfUtils<ImageType>::linearInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage,false);
    }else{
      finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage);
    }


    
    LOGV(1) << "Inverting affine" << std::endl;
    AffineTransformType::Pointer inverseAffine = AffineTransformType::New();
                                
    bool success = affine->GetInverse(inverseAffine);
    LOGV(1) << VAR(success)<<" "<<VAR(inverseAffine) << std::endl;
    if (success){
      DeformationFieldPointerType transf = TransfUtils<ImageType>::affineToDisplacementField(inverseAffine, targetImage);
      finalDeformation = TransfUtils<ImageType>::composeDeformations(finalDeformation, transf);
    }
    else{
      LOG << "Inverting affine failed, result might be incorrect" << std::endl;
    }



    //ImageUtils<DeformationFieldType>::writeImage("C:\\Users\\jstrasse\\Desktop\\a_finalDefAfter.mhd",finalDeformation);

    if (DEFORM_ATLAS_IMAGE) {
      LOG<<"Deforming Images.."<<endl;
      ImagePointerType deformedAtlasImage=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasImage,finalDeformation);
      ImageUtils<ImageType>::writeImage(filterConfig->outputDeformedFilename,deformedAtlasImage);
      LOGV(20)<<"Final SAD: "<<ImageUtils<ImageType>::sumAbsDist((ImageConstPointerType)deformedAtlasImage,(ImageConstPointerType)targetImage)<<endl;
    }


    if (originalAtlasSegmentation.IsNotNull()){
      //ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasSegmentation,finalDeformation,true);
      //ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedSegmentationFilename,deformedAtlasSegmentation);
    }  
  }

  *fieldResX = targetResX;
  *fieldResY = targetResY;
  *fieldResZ = targetResZ;
  LOG << "fieldResX " << *fieldResX << std::endl;
  LOG << "fieldResY " << *fieldResY << std::endl;
  LOG << "fieldResZ " << *fieldResZ << std::endl;

  ImageType::SizeType resultSize = finalDeformation->GetLargestPossibleRegion().GetSize();
  *fieldSizeX = resultSize[0];
  *fieldSizeY = resultSize[1];
  *fieldSizeZ = resultSize[2];
  LOG << "fieldSizeX " << *fieldSizeX << std::endl;
  LOG << "fieldSizeY " << *fieldSizeY << std::endl;
  LOG << "fieldSizeZ " << *fieldSizeZ << std::endl;

  *fieldOriginX = voiOriginX;
  *fieldOriginY = voiOriginY;
  *fieldOriginZ = voiOriginZ;
  LOG << "fieldOriginX " << *fieldOriginX << std::endl;
  LOG << "fieldOriginY " << *fieldOriginY << std::endl;
  LOG << "fieldOriginZ " << *fieldOriginZ << std::endl;

  *fieldVectors = new float[*fieldSizeX * *fieldSizeY * *fieldSizeZ * 3];

  DeformationFieldType::IndexType vectorIndex;
  int offsetStart = 0;
  for (int zt = 0; zt < *fieldSizeZ; zt++) {
    for (int yt = 0; yt < *fieldSizeY; yt++) {
      for (int xt = 0; xt < *fieldSizeX; xt++) {
        vectorIndex[0] = xt;
        vectorIndex[1] = yt;
        vectorIndex[2] = zt;
        DisplacementType a = finalDeformation->GetPixel(vectorIndex);
        offsetStart = zt * *fieldSizeY * *fieldSizeX * 3 + yt * *fieldSizeX * 3 + xt * 3;
        (*fieldVectors)[offsetStart + 0] = a[0];
        (*fieldVectors)[offsetStart + 1] = a[1];
        (*fieldVectors)[offsetStart + 2] = a[2];
      }
    }
  }
  progress = 99;
  realProgressCallbackFunc(progress);

  OUTPUTTIMER;
  if (filterConfig->logFileName!=""){
    mylog.flushLog(filterConfig->logFileName);
  }

  std::cout.rdbuf(coutbuf); //reset to standard output again


  progress = 101;
  realProgressCallbackFunc(progress);

  return NULL;

}
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

//////////////////////////////////////////////////
// The main registration algorithm entry function
//////////////////////////////////////////////////
EXTERN_C __declspec(dllexport) wchar_t* __cdecl DoRegistration2(
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
    //feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    SRSConfig filterConfig;
    //filterConfig.parseParams(argc,argv);
    if (filterConfig.logFileName!=""){
        mylog.setCachedLogging();
    }
    logSetStage("Init");
  
    typedef unsigned short PixelType;
    const unsigned int D=3;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
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
    //typedef FastUnaryPotentialRegistrationSAD< LabelMapperType, ImageType > RegistrationUnaryPotentialType;
    typedef FastUnaryPotentialRegistrationNCC< ImageType > RegistrationUnaryPotentialType;
  
    typedef PairwisePotentialRegistration< ImageType > RegistrationPairwisePotentialType;
    
    typedef PairwisePotentialCoherence< ImageType > CoherencePairwisePotentialType;
    //typedef PairwisePotentialMultilabelCoherence< ImageType > CoherencePairwisePotentialType;

  
#define POTENTIALINHERITANCE
#ifdef POTENTIALINHERITANCE
    typedef FastGraphModel<ImageType>        GraphType;
#else
    typedef FastGraphModel<ImageType,RegistrationUnaryPotentialType,RegistrationPairwisePotentialType,SegmentationUnaryPotentialType,SegmentationPairwisePotentialType,CoherencePairwisePotentialType>        GraphType;
#endif
    
    typedef HierarchicalSRSImageToImageFilter<GraphType>        FilterType;    
    //create filter
    FilterType::Pointer filter=FilterType::New();
    filter->setConfig(&filterConfig);
    
    logSetStage("Instantiate Potentials");
    

    RegistrationUnaryPotentialType::Pointer unaryRegistrationPot=RegistrationUnaryPotentialType::New();
    SegmentationUnaryPotentialType::Pointer unarySegmentationPot=SegmentationUnaryPotentialType::New();
    RegistrationPairwisePotentialType::Pointer pairwiseRegistrationPot=RegistrationPairwisePotentialType::New();
    SegmentationPairwisePotentialType::Pointer pairwiseSegmentationPot=SegmentationPairwisePotentialType::New();
    CoherencePairwisePotentialType::Pointer pairwiseCoherencePot=CoherencePairwisePotentialType::New();
#ifdef POTENTIALINHERITANCE

    filter->setUnaryRegistrationPotentialFunction(static_cast<FastUnaryPotentialRegistrationNCC<ImageType>::Pointer>(unaryRegistrationPot));
    filter->setPairwiseRegistrationPotentialFunction(static_cast< PairwisePotentialRegistration<ImageType>::Pointer>(pairwiseRegistrationPot));
    filter->setUnarySegmentationPotentialFunction(static_cast< UnaryPotentialSegmentation<ImageType>::Pointer>(unarySegmentationPot));
    filter->setPairwiseCoherencePotentialFunction(static_cast< PairwisePotentialCoherence<ImageType>::Pointer>(pairwiseCoherencePot));
    filter->setPairwiseSegmentationPotentialFunction(static_cast< PairwisePotentialSegmentation<ImageType>::Pointer>(pairwiseSegmentationPot));
#else
    filter->setUnaryRegistrationPotentialFunction((unaryRegistrationPot));
    filter->setPairwiseRegistrationPotentialFunction((pairwiseRegistrationPot));
    filter->setUnarySegmentationPotentialFunction((unarySegmentationPot));
    filter->setPairwiseCoherencePotentialFunction((pairwiseCoherencePot));
    filter->setPairwiseSegmentationPotentialFunction((pairwiseSegmentationPot));

#endif

    logUpdateStage("IO");
    logSetVerbosity(filterConfig.verbose);
    //LOG<<"Loading target image :"<<filterConfig.targetFilename<<std::endl;
    //ImagePointerType targetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);
    LOG << "create target image" << std::endl;
    ImageType::Pointer targetImage = ImageType::New();

    ImageType::SizeType  targetSize;
    targetSize[0]  = targetSizeX;  // size along X
    targetSize[1]  = targetSizeY;  // size along Y
    targetSize[2]  = targetSizeZ;  // size along Z

    ImageType::IndexType start;
    start[0] =   0;  // first index on X
    start[1] =   0;  // first index on Y
    start[2] =   0;  // first index on Z

    ImageType::RegionType targetRegion;
    targetRegion.SetSize( targetSize );
    targetRegion.SetIndex( start );

    targetImage->SetRegions(targetRegion);
    targetImage->Allocate();

    ImageType::IndexType index;
    for (int zt = 0; zt < targetSizeZ; zt++) {
      for (int yt = 0; yt < targetSizeY; yt++) {
        for (int xt = 0; xt < targetSizeX; xt++) {
          index[0] = xt;
          index[1] = yt;
          index[2] = zt;
          targetImage->SetPixel(index, targetPixels[xt + yt*targetSizeY + zt * (targetSizeX * targetSizeY)]);
        }
      }
    }
    double targetSpacing[3];
    targetSpacing[0] = targetResX;
    targetSpacing[1] = targetResY;
    targetSpacing[2] = targetResZ;
    targetImage ->SetSpacing(targetSpacing);

    //write image as a test
    typedef  itk::ImageFileWriter< ImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    std::string seriesFormat("C:\\Users\\jstrasse\\Desktop");
    seriesFormat = seriesFormat + "\\" + "target.nii.gz";
    writer->SetFileName(seriesFormat);
    writer->SetInput(targetImage);
    writer->Update();


    #if 0
      if (filterConfig.normalizeImages){
          targetImage=FilterUtils<ImageType>::normalizeImage(targetImage);
      }
    #endif

    if (!targetImage) {LOG<<"failed!"<<endl; exit(0);}


    LOG<<"create atlas image :"<<filterConfig.atlasFilename<<std::endl;
    ImagePointerType atlasImage = ImageType::New();

    ImageType::SizeType  size;
    size[0]  = sourceSizeX;  // size along X
    size[1]  = sourceSizeY;  // size along Y
    size[2]  = sourceSizeZ;  // size along Z

    ImageType::RegionType region;
    region.SetSize( size );
    region.SetIndex( start );

    atlasImage->SetRegions( region );
    atlasImage->Allocate();
    for (int zt = 0; zt < sourceSizeZ; zt++) {
      for (int yt = 0; yt < sourceSizeY; yt++) {
        for (int xt = 0; xt < sourceSizeX; xt++) {
          index[0] = xt;
          index[1] = yt;
          index[2] = zt;
          atlasImage->SetPixel(index, sourcePixels[xt + yt*sourceSizeY + zt * (sourceSizeX * sourceSizeY)]);
        }
      }
    }

    double spacing[3];
    spacing[0] = sourceResX;
    spacing[1] = sourceResY;
    spacing[2] = sourceResZ;
    atlasImage ->SetSpacing(spacing);


    //write image as a test
    WriterType::Pointer atlasWriter = WriterType::New();
    std::string seriesFormatAtlas("C:\\Users\\jstrasse\\Desktop");
    seriesFormatAtlas = seriesFormatAtlas + "\\" + "target2.nii.gz";
    atlasWriter->SetFileName(seriesFormatAtlas);
    atlasWriter->SetInput(atlasImage);
    atlasWriter->Update();





    //if (filterConfig.atlasFilename!="") {
    //  atlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
    //  #if 0
    //    if (filterConfig.normalizeImages){
    //      atlasImage=FilterUtils<ImageType>::normalizeImage(atlasImage);
    //    }
    //  #endif
    //}
    if (!atlasImage) {LOG<<"Warning: no atlas image loaded!"<<endl;


    LOG<<"Loading atlas segmentation image :"<<filterConfig.atlasSegmentationFilename<<std::endl;}
    ImagePointerType atlasSegmentation;
    if (filterConfig.atlasSegmentationFilename !="")atlasSegmentation=ImageUtils<ImageType>::readImage(filterConfig.atlasSegmentationFilename);
    if (!atlasSegmentation) {LOG<<"Warning: no atlas segmentation loaded!"<<endl; }
    
    ImagePointerType targetAnatomyPrior;
    if (filterConfig.targetAnatomyPriorFilename !="") {
      targetAnatomyPrior=ImageUtils<ImageType>::readImage(filterConfig.targetAnatomyPriorFilename);
      filterConfig.useTargetAnatomyPrior=true;
    }

    ImagePointerType atlasMaskImage=NULL;
    if (filterConfig.atlasMaskFilename!="") atlasMaskImage=ImageUtils<ImageType>::readImage(filterConfig.atlasMaskFilename);

    logResetStage;
    logSetStage("Preprocessing");

    if (filterConfig.histNorm){
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
    if (filterConfig.segment){
      if (filterConfig.targetGradientFilename!=""){
        targetGradient=(ImageUtils<ImageType>::readImage(filterConfig.targetGradientFilename));
      }else{
        targetGradient=Preprocessing<ImageType>::computeSheetness(targetImage);
        LOGI(8,ImageUtils<ImageType>::writeImage("targetsheetness.nii",targetGradient));  
      }
      if (filterConfig.atlasGradientFilename!=""){
        atlasGradient=(ImageUtils<ImageType>::readImage(filterConfig.atlasGradientFilename));
      }else{
        if (atlasImage.IsNotNull()){
          atlasGradient=Preprocessing<ImageType>::computeSheetness(atlasImage);
          LOGI(8,ImageUtils<ImageType>::writeImage("atlassheetness.nii",atlasGradient));
        }
      }
  
      if (filterConfig.useTargetAnatomyPrior && ! targetAnatomyPrior.IsNotNull() ){
        //targetAnatomyPrior=Preprocessing<ImageType>::computeSoftTargetAnatomyEstimate(targetImage);
        LOG<<"NOT YET IMPLEMENTED: Preprocessing<ImageType>::computeSoftTargetAnatomyEstimate"<<endl;
        exit(0);
      }
      //preprocessing 2: multilabel
      if (filterConfig.computeMultilabelAtlasSegmentation){
        atlasSegmentation=FilterUtils<ImageType>::computeMultilabelSegmentation(atlasSegmentation);
        filterConfig.nSegmentations=5;//TODO!!!!
      }
    }
    logResetStage;

    ImagePointerType originalTargetImage=targetImage,originalAtlasImage=atlasImage,originalAtlasSegmentation=atlasSegmentation;
    //preprocessing 3: downscaling

    if (filterConfig.downScale<1){
      double scale=filterConfig.downScale;
      LOG<<"Resampling images from "<< targetImage->GetLargestPossibleRegion().GetSize()<<" by a factor of"<<scale<<endl;
      targetImage=FilterUtils<ImageType>::LinearResample(targetImage,scale,true);
      if (atlasImage.IsNotNull()) atlasImage=FilterUtils<ImageType>::LinearResample(atlasImage,scale,true);
      if (atlasMaskImage.IsNotNull()) atlasMaskImage=FilterUtils<ImageType>::NNResample(atlasMaskImage,scale,false);
      if (atlasSegmentation.IsNotNull()) {
        atlasSegmentation=FilterUtils<ImageType>::NNResample((atlasSegmentation),scale,false);
        //ImageUtils<ImageType>::writeImage("testA.nii",atlasSegmentation);
      }
      if (filterConfig.segment){
        LOGV(3)<<"Resampling gradient images and anatomy prior by factor of "<<scale<<endl;
        targetGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)targetGradient),scale,true);
        atlasGradient=FilterUtils<ImageType>::LinearResample(((ImageConstPointerType)atlasGradient),scale,true);
        //targetGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)targetGradient,sigma),scale);
        //atlasGradient=FilterUtils<ImageType>::NNResample(FilterUtils<ImageType>::gaussian((ImageConstPointerType)atlasGradient,sigma),scale);
        if (filterConfig.useTargetAnatomyPrior){
          targetAnatomyPrior=FilterUtils<ImageType>::NNResample((targetAnatomyPrior),scale,false);
        }
            
      }
    }
    if (filterConfig.segment){
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
    if (filterConfig.useTargetAnatomyPrior){
      filter->setTargetAnatomyPrior(targetAnatomyPrior);
    }
    logSetStage("Bulk transforms");

    if (filterConfig.affineBulkTransform!=""){
      TransfUtils<ImageType>::AffineTransformPointerType affine=TransfUtils<ImageType>::readAffine(filterConfig.affineBulkTransform);
      LOGI(8,ImageUtils<ImageType>::writeImage("def.nii",TransfUtils<ImageType>::affineDeformImage(originalAtlasImage,affine,originalTargetImage)));
      //DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,originalTargetImage);
      DeformationFieldPointerType transf=TransfUtils<ImageType>::affineToDisplacementField(affine,targetImage);
      LOGI(8,ImageUtils<ImageType>::writeImage("def2.nii",TransfUtils<ImageType>::warpImage((ImageType::ConstPointer)originalAtlasImage,transf)));
      filter->setBulkTransform(transf);
    }
    else if (filterConfig.bulkTransformationField!=""){
      filter->setBulkTransform(ImageUtils<DeformationFieldType>::readImage(filterConfig.bulkTransformationField));
    }else if (filterConfig.initWithMoments){
      //LOG<<" NOT NOT NOT Computing transform to move image centers on top of each other.."<<std::endl;
      LOG<<"initializing deformation using moments.."<<std::endl;
      DeformationFieldPointerType transf=TransfUtils<ImageType>::computeCenteringTransform(originalTargetImage,originalAtlasImage);
      filter->setBulkTransform(transf);
    }
    logResetStage;//bulk transforms
    originalTargetImage=NULL;
    originalAtlasImage=NULL;
    // compute SRS
    clock_t FULLstart = clock();
    filter->Init();
    logResetStage; //init

    filter->Update();
    logSetStage("Finalizing");
    clock_t FULLend = clock();
    float t = (float) ((double)(FULLend - FULLstart) / CLOCKS_PER_SEC);
    LOG<<"Finished computation after "<<t<<" seconds"<<std::endl;
    LOG<<"Unaries: "<<tUnary<<" Optimization: "<<tOpt<<std::endl;	
    LOG<<"Pairwise: "<<tPairwise<<std::endl;

    
    //process outputs
    ImagePointerType targetSegmentationEstimate=filter->getTargetSegmentationEstimate();
    DeformationFieldPointerType finalDeformation=filter->getFinalDeformation();
    
    delete filter;
    if (filterConfig.atlasFilename!="") originalAtlasImage=ImageUtils<ImageType>::readImage(filterConfig.atlasFilename);
    if (filterConfig.targetFilename!="") originalTargetImage=ImageUtils<ImageType>::readImage(filterConfig.targetFilename);

    //upsample?
    if (filterConfig.downScale<1){
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
      ImageUtils<ImageType>::writeImage(filterConfig.segmentationOutputFilename,targetSegmentationEstimate);
    }
    
    if (finalDeformation.IsNotNull() ) {
      //if (filterConfig.defFilename!="")
      //    ImageUtils<DeformationFieldType>::writeImage(filterConfig.defFilename,finalDeformation);
      if (filterConfig.linearDeformationInterpolation){
        finalDeformation=TransfUtils<ImageType>::linearInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage,false);
      }else{
        finalDeformation=TransfUtils<ImageType>::bSplineInterpolateDeformationField(finalDeformation,(ImageConstPointerType)originalTargetImage);
      }
        
      if (filterConfig.defFilename!="")
        ImageUtils<DeformationFieldType>::writeImage(filterConfig.defFilename,finalDeformation);
        
      LOG<<"Deforming Images.."<<endl;
      ImagePointerType deformedAtlasImage=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasImage,finalDeformation);
      ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedFilename,deformedAtlasImage);
      LOGV(20)<<"Final SAD: "<<ImageUtils<ImageType>::sumAbsDist((ImageConstPointerType)deformedAtlasImage,(ImageConstPointerType)targetImage)<<endl;
        
      if (originalAtlasSegmentation.IsNotNull()){
        ImagePointerType deformedAtlasSegmentation=TransfUtils<ImageType>::warpImage((ImageConstPointerType)originalAtlasSegmentation,finalDeformation,true);
        ImageUtils<ImageType>::writeImage(filterConfig.outputDeformedSegmentationFilename,deformedAtlasSegmentation);
      }  
    }
    
    OUTPUTTIMER;
    if (filterConfig.logFileName!=""){
      mylog.flushLog(filterConfig.logFileName);
    }
  
}

#pragma once


#include "itkImage.h"
#include "Log.h"
#include "ImageUtils.h"
#include "itkAffineTransform.h"
#include <iostream>
#include "FilterUtils.hpp"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkTransformFactoryBase.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkContinuousIndex.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineTransform.h"
#include <itkBSplineDeformableTransform.h>
#include <itkWarpImageFilter.h>
#include "itkVectorLinearInterpolateImageFunction.h"
#include <itkVectorNearestNeighborInterpolateImageFunction.h>
#include <itkVectorResampleImageFilter.h>
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkDisplacementFieldCompositionFilter.h"
#include <utility>
#include <itkWarpVectorImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include "itkFixedPointInverseDeformationFieldImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include <itkVectorResampleImageFilter.h>
#include <itkDisplacementFieldTransform.h>
#include "itkTransformFactoryBase.h"
#include "itkTransformFactory.h"
#include "itkMatrixOffsetTransformBase.h"
#include <itkDisplacementFieldJacobianDeterminantFilter.h>
#include <itkDisplacementFieldToBSplineImageFilter.h>
#include "itkConstantPadImageFilter.h"
#include "itkTranslationTransform.h"

using namespace std;

template<typename ImageType, typename PointsContainerType, typename CDisplacementPrecision=float>
class LandmarkUtils {

public:
  typedef typename PointsContainerType::Pointer  PointsContainerPointerType;

  static const int D=ImageType::ImageDimension;
  typedef  CDisplacementPrecision DisplacementPrecision;
  typedef itk::Vector<DisplacementPrecision,D> DisplacementType;
  typedef itk::Point<DisplacementPrecision, D> PointType;
  typedef itk::Image<DisplacementType,D> DeformationFieldType;
  typedef typename DeformationFieldType::Pointer DeformationFieldPointerType;
  typedef typename DeformationFieldType::ConstPointer DeformationFieldConstPointerType;

  typedef typename itk::VectorLinearInterpolateImageFunction<DeformationFieldType, double> DisplacementInterpolatorType;
  typedef typename DisplacementInterpolatorType::Pointer DisplacementInterpolatorPointerType;
    
public:

  static PointsContainerPointerType transform(PointsContainerPointerType pointSet, DeformationFieldPointerType deformation) {
    PointsContainerPointerType tranformedPoints = PointsContainerType::New();
    tranformedPoints->Reserve(pointSet->Size());
    DisplacementInterpolatorPointerType displacementFieldInterpolator = DisplacementInterpolatorType::New();
    displacementFieldInterpolator->SetInputImage(deformation);
    PointsContainerType::Iterator it = pointSet->Begin();
    PointsContainerType::Iterator tranformedIt = tranformedPoints->Begin();
    while (it!= pointSet->End()) {
      PointType point = it->Value();
      DisplacementType displacement = displacementFieldInterpolator->Evaluate(point);
      DisplacementType tranformedPoint;
      for (int i = 0; i < D; i++) {
        tranformedPoint[i] = point[i] + displacement[i];
      }
      tranformedIt->Value() = tranformedPoint;

      it++;
      tranformedIt++;
    }
    
    
    return tranformedPoints;
  }


  static void logTRE(PointsContainerPointerType deformedLandmarks, PointsContainerPointerType atlasLandmarks) {
    PointsContainerType::Iterator it = deformedLandmarks->Begin();
    PointsContainerType::Iterator atlasIt = atlasLandmarks->Begin();
    int i=0;
    while (it != deformedLandmarks->End()) {
      double localError = (it->Value() - atlasIt->Value()).GetNorm();
      LOG << "point " << i << " with error: " << localError << std::endl;
      ++i;
      it++;
      atlasIt++;
    }
  }
};

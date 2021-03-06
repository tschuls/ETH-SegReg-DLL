#include "Log.h"

#include <stdio.h>
#include <iostream>
#include "ArgumentParser.h"
#include "ImageUtils.h"
#include <itkWarpImageFilter.h>

#include "TransformationUtils.h"


using namespace std;
using namespace itk;




int main(int argc, char ** argv)
{
    

	//feraiseexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    typedef unsigned char PixelType;
    const unsigned int D=2;
    typedef Image<PixelType,D> ImageType;
    typedef ImageType::Pointer ImagePointerType;
    typedef ImageType::ConstPointer ImageConstPointerType;
    typedef Vector<float,D> LabelType;
    typedef Image<LabelType,D> LabelImageType;
    typedef LabelImageType::Pointer LabelImagePointerType;
    typedef ImageType::IndexType IndexType;
    
    LabelImagePointerType deformation1 = ImageUtils<LabelImageType>::readImage(argv[1]);

    LabelImagePointerType deformation2 = ImageUtils<LabelImageType>::readImage(argv[2]);
    
    LabelImagePointerType composedDeformation= TransfUtils<ImageType>::subtract(deformation1,deformation2) ;
    
    ImageUtils<LabelImageType>::writeImage(argv[3],composedDeformation);
    
    LOG<<" "<<argv[1]<<endl;
	return 1;
}

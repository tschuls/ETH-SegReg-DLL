#pragma once
#include "matrix.h"
#include "BaseLinearSolver.h"
#include "TransformationUtils.h"
#include "Log.h"
#include "ImageUtils.h"
#include <vector>
#include <sstream>
#include "SolveAquircGlobalDeformationNormCVariables.h"

template<class ImageType>
class AquircLocalComposedErrorSolver: public AquircLocalErrorSolver< ImageType>{
public:
    typedef typename  TransfUtils<ImageType>::DeformationFieldType DeformationFieldType;
    typedef typename  DeformationFieldType::Pointer DeformationFieldPointerType;
    typedef typename ImageUtils<ImageType>::FloatImagePointerType FloatImagePointerType;
    typedef typename ImageUtils<ImageType>::FloatImageType FloatImageType;
    typedef typename itk::ImageRegionIterator<FloatImageType> FloatImageIterator;
    typedef typename itk::ImageRegionIteratorWithIndex<DeformationFieldType> DeformationFieldIterator;
    typedef typename DeformationFieldType::PixelType DeformationType;
    typedef typename DeformationFieldType::IndexType IndexType;
    typedef typename DeformationFieldType::PointType PointType;
    typedef typename ImageType::Pointer ImagePointerType;
    typedef typename ImageType::OffsetType OffsetType;

    static const unsigned int D=ImageType::ImageDimension;
public:

    virtual void SetVariables(std::vector<string> * imageIDList, map< string, map <string, DeformationFieldPointerType> > * deformationCache, map< string, map <string, DeformationFieldPointerType> > * trueDeformations,ImagePointerType img){
        LOG<<"LOCAL  ERROR SOLVER"<<endl;

        this->m_imageIDList=imageIDList;
        this->m_deformationCache=deformationCache;
        this->m_numImages=imageIDList->size();
        this->m_nPixels=2*(*deformationCache)[(*imageIDList)[0]][(*imageIDList)[1]]->GetLargestPossibleRegion().GetNumberOfPixels( );
        this->m_nEqs= this->m_numImages*(this->m_numImages-1)*( this->m_numImages-2)*this->m_nPixels;
        this->m_nVars= this->m_numImages*(this->m_numImages-1)*this->m_nPixels;
        this->m_nNonZeroes=3*this->m_nEqs;
        this->m_trueDeformations=trueDeformations;
      

        if (img.IsNotNull()){
            this->m_regionOfInterest.SetSize(img->GetLargestPossibleRegion().GetSize());
            IndexType startIndex,nullIdx;
            nullIdx.Fill(0);
            PointType startPoint;
            img->TransformIndexToPhysicalPoint(nullIdx,startPoint);
            (*this->m_deformationCache)[(*this->m_imageIDList)[0]][(*this->m_imageIDList)[1]]->TransformPhysicalPointToIndex(startPoint,startIndex);
            this->m_regionOfInterest.SetIndex(startIndex);
        }else{
            this->m_regionOfInterest=  (*this->m_deformationCache)[(*this->m_imageIDList)[0]][(*this->m_imageIDList)[1]]->GetLargestPossibleRegion();
        }

        if (trueDeformations!=NULL)
            this->computeError(deformationCache);
    }
    
    virtual void createSystem(){

        LOG<<"Creating equation system.."<<endl;
        LOG<<VAR(this->m_numImages)<<" "<<VAR(this->m_nPixels)<<" "<<VAR(this->m_nEqs)<<" "<<VAR(this->m_nVars)<<" "<<VAR(this->m_nNonZeroes)<<endl;
        mxArray *mxX=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxY=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxV=mxCreateDoubleMatrix(this->m_nNonZeroes,1,mxREAL);
        mxArray *mxB=mxCreateDoubleMatrix(this->m_nEqs,1,mxREAL);
        
        double * x=( double *)mxGetData(mxX);
        std::fill(x,x+this->m_nNonZeroes,this->m_nEqs);
        double * y=( double *)mxGetData(mxY);
        std::fill(y,y+this->m_nNonZeroes,this->m_nVars);
        double * v=( double *)mxGetData(mxV);
        double * b=mxGetPr(mxB);
        
        LOG<<"creating"<<endl;
     


        char buffer[256+1];
        buffer[256] = '\0';
        engOutputBuffer(this->m_ep, buffer, 256);
        std::vector<double> weights=this->getCircleWeights(0.0);

        //attention matlab index convention?!?
        long int eq = 1;
        long int c=0;
        long int maxE=0;
        for (int s = 0;s<this->m_numImages;++s){                            
            int source=s;
            for (int i=0;i<this->m_numImages;++i){
                if (i!=s){
                    int intermediate=i;
                    DeformationFieldPointerType d1=(*this->m_deformationCache)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];

                    for (int t=0;t<this->m_numImages;++t){
                        if (t!=i && t!=s){
                            //define a set of 3 images
                            int target=t;
                            DeformationFieldPointerType d2=(*this->m_deformationCache)[(*this->m_imageIDList)[intermediate]][(*this->m_imageIDList)[target]];
                            DeformationFieldPointerType d3=(*this->m_deformationCache)[(*this->m_imageIDList)[target]][(*this->m_imageIDList)[source]];
                            
                            DeformationFieldPointerType hatd1,hatd2,hatd3;
                            if (this->m_trueDeformations!=NULL){
                                hatd1=(*this->m_trueDeformations)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];
                                hatd2=(*this->m_trueDeformations)[(*this->m_imageIDList)[intermediate]][(*this->m_imageIDList)[target]];
                                hatd3=(*this->m_trueDeformations)[(*this->m_imageIDList)[target]][(*this->m_imageIDList)[source]];
                            }
                            
                            //compute circle
                            DeformationFieldPointerType circle=composeDeformations(d1,d2,d3);
                            //compute norm
                            DeformationFieldIterator it(circle,circle->GetLargestPossibleRegion());
                            it.GoToBegin();
                            
                            // LOG<<VAR(dir)<<" "<<VAR(start)<<endl;
                            for (;!it.IsAtEnd();++it){
                                bool valid=true;
                                IndexType idx3=it.GetIndex(),idx2,idx1;
                                PointType pt1,pt2,pt3;
#if 1                         
                                //This is the backward assumption. circle errors are in the domain of d3, and are summed backwards
                                
                                d3->TransformIndexToPhysicalPoint(idx3,pt3);
                                pt2=pt3+d3->GetPixel(idx3);
                                //pt2=pt3+hatd3->GetPixel(idx3);
                                d2->TransformPhysicalPointToIndex(pt2,idx2);
                                // what to do when circle goes outside along the way?
                                // skip it
                                if ( !d2->GetLargestPossibleRegion().IsInside(idx2) ) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    //eq+=D;
                                    //c+=3*D;
                                    continue;
                                }
                                pt1=pt2+d2->GetPixel(idx2);
                                //pt1=pt2+hatd2->GetPixel(idx2);
                                d1->TransformPhysicalPointToIndex(pt1,idx1);
                                if ( (!d1->GetLargestPossibleRegion().IsInside(idx1) )) {
                                    LOGV(6)<<"break at "<<VAR(eq)<<" "<<VAR(c)<<" "<<VAR(idx3)<<" "<<VAR(idx2)<<endl;
                                    //eq=eq+D;
                                    //c+=3*D;
                                    continue;
                                }
#else
                                //fixed point estimation
                                idx1=idx3;
                                idx2=idx3;
                                
#endif
                                
                                double val=1;

                                //add 1 for matlab array layout
                                long int e1=edgeNum(source,intermediate,idx1)+1;
                                long int e2=edgeNum(intermediate,target,idx2)+1;
                                long int e3=edgeNum(target,source,idx3)+1;
                                if (e1<=0) {LOG<<VAR(e1)<<" ????? "<<endl;}
                                if (e2<=0) {LOG<<VAR(e2)<<endl;}
                                if (e3<=0) {LOG<<VAR(e3)<<endl; }
                                //LOG<<VAR(e1)<<" "<<VAR(e2)<<" "<<VAR(e3)<<endl;
                                
                                DeformationType localDef=it.Get();
                                
                                PointType pt0;
                                pt0=pt1+d1->GetPixel(idx1);
                                
                                LOGV(4)<<"consistency check : "<<VAR(localDef)<<" ?= "<<VAR(pt0-pt3)<<endl;
                                
                                
                                for (unsigned int d=0;d<D;++d){
                                    double def=localDef[d];
                                    LOGV(6)<<VAR(e1)<<" "<<VAR(e2)<<" "<<VAR(e3)<<endl;
                                    
                                    //set sparse entries
                                    x[c]=eq;
                                    y[c]=e1+d;
                                    //LOG<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(p)<<" "<<VAR(edgeNum(source,intermediate,p))<<endl;
                                    v[c++]=weights[0];
                                    x[c]=eq;
                                    y[c]=e2+d;
                                    v[c++]=weights[1];
                                    x[c]=eq;
                                    y[c]=e3+d;
                                    v[c++]=weights[2];
                                    
                                    //set rhs
                                    b[eq-1]=def;
                                    ++eq;
                                    LOGV(6)<<"did it"<<endl;
                                }// D
                            }//image

                        }//if
                    }//target
                    if (this->m_regWeight>0.0){
                        DeformationFieldPointerType defSourceInterm=(*this->m_deformationCache)[(*this->m_imageIDList)[source]][(*this->m_imageIDList)[intermediate]];
                        DeformationFieldIterator it(defSourceInterm,defSourceInterm->GetLargestPossibleRegion());
                        it.GoToBegin();
                        for (;!it.IsAtEnd();++it){
                            DeformationType localDef=it.Get();
                            IndexType idx=it.GetIndex();
                            long int e=edgeNum(source,intermediate,idx)+1;
                            for (int n=0;n<D;++n){
                                OffsetType off;
                                off.Fill(0);
                                off[n]=1;
                                IndexType neighborIndex=idx+off;
                                if (defSourceInterm->GetLargestPossibleRegion().IsInside(neighborIndex)){
                                    long int eNeighbor=edgeNum(source,intermediate,neighborIndex)+1;
                                    DeformationType neighborDef=defSourceInterm->GetPixel(neighborIndex);
                                    LOGV(6)<<""<<VAR(idx)<<" "<<VAR(neighborIndex)<<" "<<VAR(e)<<" "<<VAR(eNeighbor)<<" "<<endl;
                                    for (unsigned int d=0;d<D;++d){
                                        LOGV(7)<<"regularizing... "<<VAR(source)<<" "<<VAR(intermediate)<<" "<<VAR(eq)<<" "<<VAR(c+3)<<" "<<endl;
                                        double def=localDef[d];
                                        double defNeighbor=neighborDef[d];
                                        x[c]=eq;
                                        y[c]=e+d;
                                        v[c++]=-this->m_regWeight;
                                        x[c]=eq;
                                        y[c]=eNeighbor+d;
                                        v[c++]=this->m_regWeight;
                                        b[eq-1]=this->m_regWeight*(defNeighbor-def);
                                        ++eq;
                                    }
                                }//inside

                            }//neighbors
                        }//for
                    }//regularization
                }//if
            }//intermediate
        }//source
        LOG<<VAR(eq)<<" "<<VAR(c)<<endl;
        this->m_nNonZeroes=c;
        mxSetM(mxX,c);
        mxSetM(mxY,c);
        mxSetM(mxV,c);
        mxSetM(mxB,eq-1);
        
        //put variables into workspace and immediately destroy them
        engPutVariable(this->m_ep,"xCord",mxX);
        mxDestroyArray(mxX);
        engPutVariable(this->m_ep,"yCord",mxY);
        mxDestroyArray(mxY);
        engPutVariable(this->m_ep,"val",mxV);
        mxDestroyArray(mxV);
        engPutVariable(this->m_ep,"b",mxB);
        mxDestroyArray(mxB);
        engEvalString(this->m_ep,"A=sparse(xCord,yCord,val);" );
        //clear unnneeded variables from matlab workspace
        engEvalString(this->m_ep,"clear xCord yCord val;" );

    }

    virtual void storeResult(string directory, string method){
        std::vector<double> result(this->m_nVars);
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<this->m_numImages;++s){
            for (int t=0;t<this->m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedError=ImageUtils<DeformationFieldType>::createEmpty((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]);
                    DeformationFieldIterator it(estimatedError,estimatedError->GetLargestPossibleRegion());
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it){
                        DeformationType disp;
                        int e=edgeNum(s,t,it.GetIndex());
                        for (unsigned int d=0;d<D;++d,++p){
                            disp[d]=rData[e+d];
                        }
                        it.Set(disp);
                    }

                    ostringstream outfile;
                    outfile<<directory<<"/estimatedError-"<<method<<"-FROM-"<<(*this->m_imageIDList)[s]<<"-TO-"<<(*this->m_imageIDList)[t]<<".mha";
                    ImageUtils<DeformationFieldType>::writeImage(outfile.str().c_str(),estimatedError);
                }
            }
        }

    }

     virtual map< string, map <string, DeformationFieldPointerType> > * getEstimatedDeformations(){
        map< string, map <string, DeformationFieldPointerType> > * result=new map< string, map <string, DeformationFieldPointerType> >;
        double * rData=mxGetPr(this->m_result);
        for (int s = 0;s<this->m_numImages;++s){
            for (int t=0;t<this->m_numImages;++t){
                if (s!=t){
                    //slightly(!!!) stupid creation of empty image
                    DeformationFieldPointerType estimatedDef=ImageUtils<DeformationFieldType>::createEmpty((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]);
                    DeformationFieldIterator it(estimatedDef,estimatedDef->GetLargestPossibleRegion());
                    DeformationFieldIterator itOriginalDef((*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]],(*this->m_deformationCache)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]->GetLargestPossibleRegion());
                    itOriginalDef.GoToBegin();
                    it.GoToBegin();
                    for (int p=0;!it.IsAtEnd();++it,++itOriginalDef){
                        DeformationType disp;
                        int e=edgeNum(s,t,it.GetIndex());

                        for (unsigned int d=0;d<D;++d,++p){
                            disp[d]=rData[e+d];
                        }
                        it.Set(itOriginalDef.Get()-disp);
                    }

                    (*result)[(*this->m_imageIDList)[s]][(*this->m_imageIDList)[t]]=estimatedDef;
                }
            }
        }
        return result;
    }

};

#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

#define Melloc3X3Uchar(t) {t=new uchar*[3];for(int i=0;i<3;i++){t[i]=new uchar[3];}}
#define Free3X3Uchar(t) {for(int i=0;i<3;i++)delete[] t[i];}

#define MellocnXnDouble(t,n) {t=new double*[n];for(int i=0;i<n;i++){t[i]=new double[n];}}
#define FreenXnarray(t,n) {for(int i=0;i<n;i++)delete[] t[i];}

#define PI 3.141592653

#define CPU_COREs 8

using namespace std;

void array3X32Mat_freeArray(cv::Mat& temp,uchar** t){
    temp.create(3,3,CV_8UC1);
    for(int r=0;r<temp.rows;r++){
        uchar* curr = temp.ptr(r);
        for(int l=0;l<temp.cols;l++){
            curr[l]=t[r][l];
        }
    }
    Free3X3Uchar(t)
}

void arraynXn2Mat_freeArray(cv::Mat& temp,double** t,int n){
    temp.create(n,n,CV_64FC1);
    for(int r=0;r<temp.rows;r++){
        double* curr = temp.ptr<double>(r);
        for(int l=0;l<temp.cols;l++){
            curr[l]=t[r][l];
        }
    }
    FreenXnarray(t,n)
}

void Mat2array3X3(const cv::Mat& temp,uchar** t){
    Melloc3X3Uchar(t)
    for(int r=0;r<temp.rows;r++){
        const uchar* curr = temp.ptr(r);
        for(int l=0;l<temp.cols;l++){
            t[r][l]=curr[l];
        }
    }
}

void Mat2array3X3(const cv::Mat& temp,double** t){
    MellocnXnDouble(t,3)
    for(int r=0;r<temp.rows;r++){
        const double* curr = temp.ptr<double>(r);
        for(int l=0;l<temp.cols;l++){
            t[r][l]=curr[l];
        }
    }
}

cv::Mat convolution(const cv::Mat& source ,cv::Mat temp){
    cv::Mat result;
    cv::Mat grayImg;
    if (source.channels() == 3) {
        cv::cvtColor(source, grayImg, CV_BGR2GRAY);
    }
    else {
        source.copyTo(grayImg);
    }
    double** t;
    Mat2array3X3(temp,t);

    std::cout<<"begin convolution Mat"<<std::endl;
    result.create(grayImg.rows-2,grayImg.cols-2,grayImg.type());
    omp_set_num_threads(CPU_COREs);
#pragma omp parallel for
    for(int r=1;r<grayImg.rows-1;r++){
        const uchar* prev = grayImg.ptr(r - 1);
        const uchar* curr = grayImg.ptr(r);
        const uchar* next = grayImg.ptr(r + 1);
        uchar* pdst = result.ptr(r);
        for(int l=1;l<grayImg.cols-1;l++){
            pdst[l-1]=prev[l-1]*t[0][0]+prev[l]*t[0][1]+prev[l+1]*t[0][2]+\
                      curr[l-1]*t[1][0]+curr[l]*t[1][1]+curr[l+1]*t[1][2]+\
                      next[l-1]*t[2][0]+next[l]*t[2][1]+next[l+1]*t[2][2];
        }
    }
    Free3X3Uchar(t)
    return result;
}

cv::Mat convolution(const cv::Mat& source ,double t[3][3]){
    cv::Mat result;
    cv::Mat grayImg;
    if (source.channels() == 3) {
        cv::cvtColor(source, grayImg, CV_BGR2GRAY);
    }
    else {
        source.copyTo(grayImg);
    }
    std::cout<<"begin convolution double sizeof(double)"<<sizeof(double)<<std::endl;
    result.create(grayImg.rows-2,grayImg.cols-2,grayImg.type());
    omp_set_num_threads(CPU_COREs);
#pragma omp parallel for
    for(int r=1;r<grayImg.rows-1;r++){
        const double* prev = grayImg.ptr<double>(r - 1);
        const double* curr = grayImg.ptr<double>(r);
        const double* next = grayImg.ptr<double>(r + 1);
        double* pdst = result.ptr<double>(r);
        for(int l=1;l<grayImg.cols-1;l++){
            pdst[l-1]=prev[l-1]*t[0][0]+prev[l]*t[0][1]+prev[l+1]*t[0][2]+\
                      curr[l-1]*t[1][0]+curr[l]*t[1][1]+curr[l+1]*t[1][2]+\
                      next[l-1]*t[2][0]+next[l]*t[2][1]+next[l+1]*t[2][2];
        }
    }
    return result;
}


void getGuassCannyKernel(cv::Mat& Kx,cv::Mat& Ky,double sigma){
    double** gK;
    MellocnXnDouble(gK,5)
    double sigma_2 = sigma * sigma;
    double sum = 0;
    std::cout<<"begin calculate Kernel"<<std::endl;
    omp_set_num_threads(CPU_COREs);
#pragma omp parallel for
    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
            gK[i][j]=exp(-((i-2)*(i-2)+(j-2)*(j-2))/(2*sigma_2))/(2*PI*sigma_2);
            sum+=gK[i][j];
        }
    }
    omp_set_num_threads(CPU_COREs);
#pragma omp parallel for
    for(int i=0;i<5;i++){
        for(int j=0;j<5;j++){
            gK[i][j]/=sum;
        }
    }
    double Cx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    double Cy[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
    std::cout<<"begin strans to Mat"<<std::endl;
    cv::Mat K;
    arraynXn2Mat_freeArray(K,gK,5);
    std::cout<<"begin convolution Canny Kernel"<<std::endl;
    Ky=convolution(K,Cy);
    Kx=convolution(K,Cx);
}

int main() {
    cv::Mat img=cv::imread("/home/dang/cancerdetect/resource/samples/neg_all/image(2).jpg");
    cv::Mat Kx,Ky;
    getGuassCannyKernel(Kx,Ky,1);
    std::cout<<"begin convolution result"<<std::endl;
    cv::Mat res=convolution(img,Kx);
    cv::Mat res2=convolution(img,Ky);
    cv::imshow("Kx",res);
    cv::waitKey(20);
    cv::imshow("Ky",res2);
    cv::waitKey(0);

    cout << "Hello, World!" << endl;
    return 0;
}
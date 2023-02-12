#ifndef PROPERTIESH
#define PROPERTIESH

/*-----------------------------------LIBRARIES-------------------------------------*/
#include "Color.h"

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Properties {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Color objectColor;      //Object color
    float kd;               //Diffuse component
    float ks;               //Specular component
    float shine;            //Phong cosine power for highlights
    float t;                //Transmittance (fraction of the transmitting ray)
    float iof;              //Index of refraction

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Properties();
    __host__ __device__ Properties(const Properties &prop);
    __host__ __device__ Properties(Color objectColor, float kd, float ks, float shine, float t, float iof);
    __host__ __device__ ~Properties() = default;
};

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ __device__ Properties::Properties() {}
__host__ __device__ Properties::Properties(const Properties &prop) { *this = prop;}
__host__ __device__ Properties::Properties(Color objectColor, float kd, float ks, float shine, float t, float iof)
    : objectColor(objectColor), kd(kd), ks(ks), shine(shine), t(t), iof(iof) {}

#endif
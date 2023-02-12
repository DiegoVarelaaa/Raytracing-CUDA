#ifndef LIGHTH
#define LIGHTH

/*-----------------------------------LIBRARIES------------------------------------*/
#include "Color.h"
#include "Vector.h"
#include <iostream>

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Light {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Color color;
    Vector position;

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ __device__ void print();

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Light();
    __host__ __device__ Light(const Light &light);
    __host__ __device__ Light(Color c, Vector p);
    __host__ __device__ ~Light() = default;
};

/*----------------------------------FUNCTIONS----------------------------------*/
__host__ __device__ void Light::print() { 
    // std::cout<<"["; position.print(); 
    // std::cout<<", "; color.print(); 
    // std::cout<<"]\n";

    printf("["); position.print();
    printf(", "); color.print(); 
    printf("] \n");
}

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ __device__ Light::Light() {};
__host__ __device__ Light::Light(const Light &light) { *this = light;}
__host__ __device__ Light::Light(Color c, Vector p) : color(c), position(p) {}


#endif
#ifndef RAYH
#define RAYH

/*-----------------------------------LIBRARIES-------------------------------------*/
#include "Vector.h"

/*-----------------------------------CONSTANTS-------------------------------------*/
#define PI 3.14159265
#define SKY 100000

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Ray {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Vector origin;
    Vector direction;
    float distance;
    Vector P;          //P = o + d(distance)

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ __device__ void print();
    __device__ void setP();
    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __device__ Ray();
    __device__ Ray(const Ray &ray);
    __device__ Ray(Vector o, Vector d, float distance);
    __device__ Ray(Vector o, Vector d);
    __device__ ~Ray() = default;
};


/*----------------------------------FUNCTIONS----------------------------------*/
__host__ __device__ void Ray::print() {
    // std::cout<<"RAY: [";
    printf("RAY: [");
    origin.print(); direction.print(); P.print(); 
    // std::cout<<"]"<<std::endl;
    printf("]");
}

__device__ void Ray::setP() { P = origin + direction*(distance); }

/*--------------------------------CONSTRUCTORS---------------------------------*/
__device__ Ray::Ray() { distance = SKY;}
__device__ Ray::Ray(const Ray &ray) { *this = ray; }
__device__ Ray::Ray(Vector o, Vector d, float distance) : 
         origin(o), direction(d), distance(distance) { 
    P = origin + direction*(distance);
}
__device__ Ray::Ray(Vector o, Vector d) : origin(o) {
    distance = SKY; 
    Vector v(d,o); v.normalize();
    this->direction = v; 
}


#endif
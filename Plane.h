#ifndef PLANEH
#define PLANEH

/*-----------------------------------LIBRARIES------------------------------------*/
#include "Properties.h"
#include "Primitive.h"
#include "Vector.h"
#include "Ray.h"

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Plane: public Primitive {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Vector c;           //Point
    Vector v;           //Normal plane

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ __device__ void print();
    //Find intersection between a ray and Plane
    __device__ void rayIntersection(Ray *ray, Primitive **object);
    //Get normal given an intersection point
    __device__ Vector getNormal(Vector P);
    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Plane();
    __host__ __device__ Plane(Vector c, Vector v, Properties p);
    __host__ __device__ Plane(Vector c, Vector v);
    __host__ __device__ Plane(const Plane &pln);
    __host__ __device__ ~Plane() = default;
};


/*----------------------------------FUNCTIONS----------------------------------*/

__device__ void Plane::rayIntersection(Ray *ray, Primitive **object) {
    Vector voc = (ray->origin) - (c);
    float dv = (ray->direction).dotPoint(v);
    float vocv = voc.dotPoint(v); 
    if(dv != 0) {
        ray->distance = -(vocv)/dv;
        if(ray->distance < 0)
            ray->distance = SKY;
        else {
            *object = this;
            ray->setP();
        }
    }
}

__device__ Vector Plane::getNormal(Vector P) { return v; }

__host__ __device__ void Plane::print(){ 
    // std::cout<<"Plane ct["; c.print();
    // std::cout<<"], v["; v.print();
    // std::cout<<"]"<<std::endl;

    printf("Plane ct[");c.print();
    printf("], v[");v.print();
    printf("] \n");
}

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ __device__ Plane::Plane() {};
__host__ __device__ Plane::Plane(Vector c,Vector v, Properties p) : c(c), v(v) { this->properties = p;}
__host__ __device__ Plane::Plane(Vector c,Vector v) : c(c), v(v) {}
__host__ __device__ Plane::Plane(const Plane &cln) { *this = cln; }

#endif
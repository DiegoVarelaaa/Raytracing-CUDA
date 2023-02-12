#ifndef SPHEREH
#define SPHEREH

/*-----------------------------------LIBRARIES------------------------------------*/
#include "Properties.h"
#include "Primitive.h"
#include "Vector.h"
#include "Ray.h"

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Sphere: public Primitive {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Vector center;      //Center
    float r;            //Ratio

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ __device__ void print();
    //Find intersection betweern a ray and sphere
    __device__ void rayIntersection(Ray *ray, Primitive **object);
    //Get normal given an intersection point
    __device__ Vector getNormal(Vector P);

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Sphere();
    __host__ __device__ Sphere(Vector c, float r, Properties p);
    __host__ __device__ Sphere(const Sphere &sph);
    __host__ __device__ ~Sphere() = default;
};


/*----------------------------------FUNCTIONS----------------------------------*/
__device__ void Sphere::rayIntersection(Ray *ray, Primitive **object) {
    float a, b, c;
    Vector voc = (ray->origin) - (center);
    a = 1;
    b = 2*(ray->direction).dotPoint(voc);
    c = voc.dotPoint(voc) - r*r;
    quadraticSolution(a, b, c, &(ray->distance));
    if(ray->distance != SKY){ 
        ray->setP(); 
        *object = this;
    }
}


__device__ Vector Sphere::getNormal(Vector P) {
    Vector N(P,center); 
    N.normalize();
    return N;
}

__host__ __device__ void Sphere::print() { 
    printf("Sphere [");center.print(); 
    printf(", %f] \n", r);
    }

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ __device__ Sphere::Sphere() {};
__host__ __device__ Sphere::Sphere(Vector c, float r, Properties p): center(c), r(r) { this->properties = p;}
__host__ __device__ Sphere::Sphere(const Sphere &sph) { *this = sph; }

#endif
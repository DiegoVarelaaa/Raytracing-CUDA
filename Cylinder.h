#ifndef CYLINDERH
#define CYLINDERH

/*-----------------------------------LIBRARIES------------------------------------*/
#include "Properties.h"
#include "Primitive.h"
#include "Vector.h"
#include "Ray.h"

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Cylinder: public Primitive {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Vector ct;          //Top cap
    Vector cb;          //Botton cap
    float r;            //Radio
    Vector v;           //Axys direction
    float maxm;         //Cilynder 
    float m;            //Closest point to intersection point over axys direction

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ __device__ void print();
    //Find intersection between a ray and Cilynder
    __device__ void rayIntersection(Ray *ray, Primitive **object);
    //Get normal given an intersection point
    __device__ Vector getNormal(Vector P);

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Cylinder();
    __host__ __device__ Cylinder(Vector ct, Vector cb, float r, Properties p);
    __host__ __device__ Cylinder(const Cylinder &cln);
    __host__ __device__ ~Cylinder() = default;
};


/*----------------------------------FUNCTIONS----------------------------------*/
__device__ void Cylinder::rayIntersection(Ray *ray, Primitive **object) {
    float a, b, c, RV;
    RV = (ray->direction).dotPoint(v);
    Vector Vocb = (ray->origin) - cb;
    float VocbDotv = Vocb.dotPoint(v);
    a = 1 - RV*RV;
    b = 2*((ray->direction).dotPoint(Vocb) - RV*VocbDotv);
    c = Vocb.dotPoint(Vocb) - VocbDotv*VocbDotv - r*r;
    quadraticSolution(a, b, c, &(ray->distance));

    if(ray->distance != SKY) {
        m = ((ray->direction)*(ray->distance)).dotPoint(v) + VocbDotv;
        if(m>0 && m<maxm) {
            *object = this;
            ray->setP();
        }
        else
            ray->distance = SKY;
    }
}

__device__ Vector Cylinder::getNormal(Vector P) {
    Vector cp = cb + v*m;
    Vector N(P, cp);
    N.normalize();
    return N;
}

__host__ __device__ void Cylinder::print() { 
    // std::cout<<"Cylinder ct["; ct.print();
    // std::cout<<"], cb["; cb.print();
    // std::cout<<"], r "<<r<<std::endl;

    printf("Cylinder ct");ct.print();
    printf(", cb");cb.print();
    printf(", r [%f]", r);printf("\n");
}

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ __device__ Cylinder::Cylinder() {};
__host__ __device__ Cylinder::Cylinder(Vector ct,Vector cb, float r, Properties p) : 
                    ct(ct), cb(cb), r(r){    
    Vector v = (ct - cb); 
    this->maxm = v.lenght();
    v.normalize(); this->v = v;
    this->m = 0;
    this->properties = p;
}
__host__ __device__ Cylinder::Cylinder(const Cylinder &cln) {*this = cln;}


#endif
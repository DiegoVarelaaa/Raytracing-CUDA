#ifndef VECTORH
#define VECTORH

/*-----------------------------------LIBRARIES-------------------------------------*/
#include <math.h>
#include <iostream>

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Vector {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    float x,y,z;

    /*----------------------------------FUNCTIONS----------------------------------*/
    //Print info
//     __host__ __device__ void print() { std::cout<<"["<<x<<","<<y<<","<<z<<"]";}
    __host__ __device__ void print() { printf("[%f,%f,%f]",x,y,z);}

    //Vector lenght
    __host__ __device__ float lenght() { return sqrt(x*x + y*y + z*z); }
    
    //Normalize info
    __host__ __device__ void normalize() { *this = *this/lenght(); }

    //Dot
    __host__ __device__ float dotPoint(Vector v) const
            { return ((x*v.x)+(y*v.y)+(z*v.z));}
    //Scale
    __host__ __device__ Vector operator* (const float &v) const
            { return Vector((x*v),(y*v),(z*v));}
    //Divide
    __host__ __device__ Vector operator/ (const float &v) const
            { return Vector((x/v),(y/v),(z/v));}
    //Add
    __host__ __device__ Vector operator+ (const Vector &v) const
            { return Vector((x+v.x),(y+v.y),(z+v.z));}
    //Substract
    __host__ __device__ Vector operator- (const Vector &v) const
            { return Vector((x-v.x),(y-v.y),(z-v.z));}
    //Cross
    __host__ __device__ Vector cross(Vector v) const
            { return Vector(((y*v.z)-(z*v.y)),((z*v.x)-(x*v.z)),((x*v.y)-(y*v.x)));}
    //Equal than
    __host__ __device__ int operator== (const Vector &v) const
            { return ((x==v.x) && (y==v.y) && (z==v.z)); }
    //Less than
    __host__ __device__ int operator< (const Vector &v) const
            { return ((x<v.x) && (y<v.y) && (z<v.z)); }
    //Less or equal
    __host__ __device__ int operator<= (const Vector &v) const
            { return ((x<=v.x) && (y<=v.y) && (z<=v.z)); }
    //greater than
    __host__ __device__ int operator> (const Vector &v) const
            { return ((x>v.x) && (y>v.y) && (z>v.z)); }
    //greater or equal
    __host__ __device__ int operator>= (const Vector &v) const
            { return ((x>=v.x) && (y>=v.y) && (z>=v.z)); }
    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Vector() { x = 0; y = 0; z = 0; }
    __host__ __device__ Vector(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ Vector(const Vector &v) { *this = v;}
    __host__ __device__ Vector(Vector a, Vector b) { *this = a - b;}
    __host__ __device__ ~Vector() = default;
};

#endif
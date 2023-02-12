#ifndef COLORH
#define COLORH

/*-----------------------------------LIBRARIES------------------------------------*/
#include <iostream>

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Color {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    float R, G, B;

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ __device__ void print();
    //Add
    __host__ __device__ Color operator+ (const Color &c) const;
    //Substract
    __host__ __device__ Color operator- (const Color &c) const;
    //Scale
    __host__ __device__ Color operator* (const float &v) const;

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ __device__ Color();
    __host__ __device__ Color(float R, float G, float B);
    __host__ __device__ Color(const Color &color);
    __host__ __device__ ~Color() = default;
};

/*----------------------------------FUNCTIONS----------------------------------*/
//Print info
// __host__ __device__ void Color::print() { std::cout<<"["<<R<<" "<<G<<" "<<B<<"]";}
__host__ __device__ void Color::print() { printf("[%f,%f,%f]",R,G,B);}

__host__ __device__ Color Color::operator+ (const Color &c) const
        { return Color((R + c.R), (G + c.G), (B + c.B));}

__host__ __device__ Color Color::operator- (const Color &c) const
        { return Color((R - c.R), (G - c.G), (B - c.B));}

__host__ __device__ Color Color::operator* (const float &v) const
        { return Color((R * v),(G * v),(B * v));}

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ __device__ Color::Color() { R = 0, G = 0, B = 0;}
__host__ __device__ Color::Color(float R, float G, float B) : R(R), G(G), B(B) {}
__host__ __device__ Color::Color(const Color &color) { *this = color;}

#endif
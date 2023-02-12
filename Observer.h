#ifndef OBSERVERH
#define OBSERVERH

/*-----------------------------------LIBRARIES-------------------------------------*/
#include "Vector.h"

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Observer {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Vector from;        //Eye location in XYZ.
    Vector lookAt;      //A position to be at the center of the image
    Vector up;          //A vector defining which direction is up
    float angle;        //In degrees

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ Observer();
    __host__ Observer(const Observer &obs);
    __host__ Observer(Vector from, Vector lookAt, Vector up, float angle);
    __host__ ~Observer() = default;
};

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ Observer::Observer() {};
__host__ Observer::Observer(const Observer &obs) { *this = obs; }
__host__ Observer::Observer(Vector from, Vector lookAt, Vector up, float angle) :
                   from(from), lookAt(lookAt), up(up), angle(angle) {}

#endif
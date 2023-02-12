#ifndef SCENEH
#define SCENEH

/*-----------------------------------LIBRARIES------------------------------------*/
#include "Primitive.h"
#include "Vector.h"
#include "Light.h"
#include "cudaErrors.h"

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Scene {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Color background;
    int pSize = 0; 
    int lSize = 0; 
    Primitive **primitives;
    Light *lights;

    /*----------------------------------FUNCTIONS----------------------------------*/
    __host__ void addPrimitive(Primitive *p);
    __host__ void allocPrimitives(size_t primitivesSize);
    __host__ __device__ void print();
    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ Scene();
    __host__ Scene(Color bg, Primitive** primitives, Light* lights);
    __host__ Scene(const Scene &s);
    __host__ ~Scene() = default;
};

/*----------------------------------FUNCTIONS----------------------------------*/
__host__ void Scene::allocPrimitives(size_t primitivesSize){ 
    primitives = new Primitive*[primitivesSize];
}
__host__ void Scene::addPrimitive(Primitive *p){primitives[pSize] = p; pSize++; }

__host__ __device__ void Scene::print() {
    printf("------------------------------------------------ \n");
    printf("SCENE INFO: \n");
    printf("\n-Background color [R G B]: "); background.print(); printf("\n");
    printf("\n-Objects info:");printf("\n");
    for(unsigned int i= 0; i< pSize; i++){
        primitives[i]->print();
    }
    printf("\n-Lights sources info: \n");
    for(unsigned int i= 0; i< lSize; i++)
        (lights[i]).print();
    printf("------------------------------------------------ \n");
}

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ Scene::Scene() {};
__host__ Scene::Scene(Color bg, Primitive** primitives, Light* lights) :
             background(bg), primitives(primitives), lights(lights) {}
__host__ Scene::Scene(const Scene &s) {
    background = s.background;
    primitives = s.primitives;
    lights = s.lights;
}

#endif
#ifndef RAYTRACERH
#define RAYTRACERH


/*-----------------------------------LIBRARIES-------------------------------------*/
#include <fstream>

#include "Scene.h"
#include "Screen.h"
#include "Observer.h"

struct ScreenItr{ Vector scanLine, pixelWidth, pixelHeight;};

/*---------------------------------CLASS DEFINITION--------------------------------*/
class RayTracer {
    public: 
    /*----------------------------------ATTRIBUTES---------------------------------*/
    Scene scene;
    Observer observer;
    Screen screen;
    ScreenItr sItr;
    Color *image;

    /*----------------------------------FUNCTIONS----------------------------------*/
    // __global__ void render(int xResolution, int yResolution, Color *image, Primitive *object);
    __device__ void intersectionTest(Ray *primaryRay, Primitive **object);
    __device__ Color shading(Ray ray, Primitive *object, int depth);
    __device__ Color colorContribution(Ray ray, Primitive *object, int depth);
    __device__ Color fullScale(Vector P, Vector N, Vector V, Primitive *object, int depth);
    __host__ void setScreenItr();
    __host__ void imageToPPM();

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ RayTracer();
    __host__ RayTracer(const RayTracer &s);
    __host__ RayTracer(Scene scene, Observer observer, Screen screen);
    // ~RayTracer() { delete image; };
};


/*----------------------------------FUNCTIONS----------------------------------*/

__host__ void RayTracer::imageToPPM() {
    //PPM file
    FILE *fp = fopen("Output/file.ppm", "w+");
    if (fp!=NULL) {
        fprintf(fp, "P3\n%d %d\n255\n", (int)screen.width, (int)screen.height);
        //Write pixel information
        auto pixels = screen.width * screen.height;
        for(auto i=0; i<pixels; i++)
            fprintf(fp, "%d %d %d ",(int)image[i].R, (int)image[i].G, (int)image[i].B);
        fclose (fp);
    }
    else {
        printf("'Output' folder doesn't exist\n");
    }
}

__device__ void RayTracer::intersectionTest(Ray *primaryRay, Primitive **object) {
    Primitive *obj;
    Ray ray = *primaryRay;
    unsigned int primitivesSize = scene.pSize;
    for(unsigned int i=0; i<primitivesSize; i++) { 
        (scene.primitives[i])->rayIntersection(&ray, &obj);
        if((ray.distance) < (primaryRay->distance)) {
            *primaryRay = ray; 
            *object = obj;
        }
    }
} 

__device__ Color RayTracer::shading(Ray ray, Primitive *object, int depth) {
    Color b = scene.background;
    if(ray.distance == SKY)
        return b;
    else
        // return b;
        return colorContribution(ray, object, depth);
}

__device__ Color RayTracer::colorContribution(Ray ray, Primitive *object, int depth) {
    Vector P = ray.P;
    Vector N = object->getNormal(P);
    Vector V = ray.direction*(-1);
    // return Color();
    return fullScale(P, N, V, object, depth);
}

__device__ Color RayTracer::fullScale(Vector P, Vector N, Vector V, Primitive *object, int depth) {
    //Primitive properties
    Color objectColor = (object->properties).objectColor;
    float shine = (object->properties).shine;
    float kd = (object->properties).kd;
    float ks = (object->properties).ks;
    //Aux variables
    Color color;
    Light *lights = scene.lights;
    unsigned int lightsSize = scene.lSize;
    Primitive *shadowObject;
    Vector offsetPoint;
    float diffuse, specular;
    float intensity = 1/sqrtf(lightsSize);
    /*-----------------------Light contribution-------------------------*/
    for(unsigned int i=0; i<lightsSize; i++) {
        Vector L((lights[i]).position, P); L.normalize();
        offsetPoint = P + L*(10e-4);
        /*------------------------Shadow ray----------------------------*/
        //Add a small offset to avoid the ray to hit the same sphere
        Ray shadowRay(offsetPoint, L, SKY);
        intersectionTest(&shadowRay, &shadowObject);
        /*-----------------------------Shade----------------------------*/
        if(shadowRay.distance == SKY) {
            /*-------------------Diffuse component----------------------*/
            diffuse = kd*max(float(0), float(L.dotPoint(N)));
            /*-------------------Specular component---------------------*/
            Vector R = L*(-1) + N*(2*(N.dotPoint(L)));
            R.normalize();
            specular = ks*pow(float(max(float(0), float(R.dotPoint(V)))), float(shine));
            /*-------------------Light contribution---------------------*/
            color.R += ((diffuse*objectColor.R) + specular)*(intensity);
            color.G += ((diffuse*objectColor.G) + specular)*(intensity);
            color.B += ((diffuse*objectColor.B) + specular)*(intensity);                                      
        }
    }
    if(depth<2 && ks>0) {
        Primitive *hitObject;
        Vector R = V*(-1) + N*(2*(N.dotPoint(V)));
        R.normalize();
        offsetPoint = P + R*(10e-4);
        Ray reflectedRay(offsetPoint, R);
        intersectionTest(&reflectedRay, &hitObject);
        Color reflectedColor = shading(reflectedRay, hitObject, depth + 1);
        color.R += (reflectedColor.R)*(ks);
        color.G += (reflectedColor.G)*(ks);
        color.B += (reflectedColor.B)*(ks);
    }
    return color;
}

__host__ void RayTracer::setScreenItr() {
    Vector w(observer.lookAt, observer.from); w.normalize();
    Vector u = w.cross(observer.up); u.normalize();
    Vector v = u.cross(w);
    float tanFOV = tan((observer.angle/2)*PI/180.0);
    float aspectRatio = screen.width/screen.height;
    float cameraheight = tanFOV*2;  
    float camerawidth  = aspectRatio*cameraheight;
    float pixelH = cameraheight/screen.height; sItr.pixelHeight = v*pixelH;
    float pixelW = camerawidth/screen.width;   sItr.pixelWidth = u*pixelW;         
    Vector xComponent = u; xComponent = xComponent*((screen.width*pixelW)/2);
    Vector yComponent = v; yComponent = yComponent*((screen.height*pixelH)/2);
    Vector corner = observer.from + w - xComponent + yComponent;
    sItr.scanLine =  corner - (sItr.pixelHeight)*(1/2) + (sItr.pixelWidth)*(1/2);
}

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ RayTracer::RayTracer() {};
__host__ RayTracer::RayTracer(const RayTracer &s){ *this = s;}
__host__ RayTracer::RayTracer(Scene scene, Observer observer, Screen screen) : scene(scene), observer(observer), screen(screen) {}

#endif
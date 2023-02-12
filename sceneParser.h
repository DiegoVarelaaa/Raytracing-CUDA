#ifndef SCENEPARSERH
#define SCENEPARSERH
/*-----------------------------------LIBRARIES---------------------------------*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "Scene.h"
#include "Screen.h"
#include "Observer.h"
#include "Properties.h"
#include "Raytracer.h"

#include "Cylinder.h"
#include "Plane.h"
#include "Polygon.h"
#include "Sphere.h"
#include "cudaErrors.h"

#define NEXT inputFile>>tok

/*----------------------------------FUNCTIONS----------------------------------*/
__host__ int readFile(std::string filename, RayTracer *RT, RayTracer *RT_device);
__host__ void readBackground(std::ifstream &inputFile, Scene *scene);
__host__ void readLight(std::ifstream &inputFile, Scene *scene, Light *ls_ptr);
__host__ void readCylinder(std::ifstream &inputFile, Scene *scene, Properties p, Cylinder *cy_ptr);
__host__ void readPolygon(std::ifstream &inputFile, Scene *scene, Properties p, Polygon **po_ptr);
__host__ void readSphere(std::ifstream &inputFile, Scene *scene, Properties p, Sphere *sp_ptr);
__host__ void readObserver(std::ifstream &inputFile, Observer *observer);
__host__ void readScreen (std::ifstream &inputFile, Screen *s);
__host__ Properties readProperties(std::ifstream &inputFile);
__host__ void copyToDevice(RayTracer *RT_host, RayTracer *RT_device, Sphere* sp_ptr, Cylinder* cy_ptr, int sp, int cy);

__global__ void copy_primitives(RayTracer *RT_device, Sphere* sp_info, Cylinder* cy_info, Polygon **po_info, int sp, int cy, int po){
    RT_device->scene.primitives = new Primitive*[(RT_device->scene.pSize)];
    int primitiveCounter = 0;

    Sphere *device_spheres = new Sphere[sp];
    for(int i=0; i<sp; i++){
        device_spheres[i] = Sphere((sp_info+i)->center, (sp_info+i)->r, (sp_info+i)->properties);
        RT_device->scene.primitives[primitiveCounter] = device_spheres+i;
        primitiveCounter++;
    }

    Cylinder *device_cylinders = new Cylinder[cy];
    for(int i=0; i<cy; i++){
        device_cylinders[i] = Cylinder((cy_info+i)->ct, (cy_info+i)->cb, (cy_info+i)->r, (cy_info+i)->properties);
        RT_device->scene.primitives[primitiveCounter] = device_cylinders+i;
        primitiveCounter++;
    }

    Polygon **device_polygons = new Polygon*[po];
    for(int i=0; i<po; i++){
        device_polygons[i] = new Polygon(po_info[i]->nVertices, po_info[i]->vertices, po_info[i]->properties);
        RT_device->scene.primitives[primitiveCounter] = device_polygons[i];
        primitiveCounter++;
    }

    // RT_device->scene.print();
}

/*----------------------------------FUNCTIONS----------------------------------*/
__host__ void copyToDevice(RayTracer *RT_host, RayTracer *RT_device, Sphere* sp_ptr, Cylinder* cy_ptr, Polygon** po_ptr,int sp, int cy, int po){
    //Sizes
    int lightSize = (RT_host->scene.lSize);
    //Copy lights
    Light *lights_device;
    checkCudaErrors(cudaMalloc((void **) &(lights_device), lightSize * sizeof(Light)));
    checkCudaErrors(cudaMemcpy((lights_device), (RT_host->scene.lights), lightSize * sizeof(Light), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(RT_device->scene.lights), &(lights_device), lightSize * sizeof(Light), cudaMemcpyHostToDevice));

    //Copy observer, screen, screenitr
    checkCudaErrors(cudaMemcpy(&(RT_device->observer), &(RT_host->observer), sizeof(Observer), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(RT_device->screen), &(RT_host->screen), sizeof(Screen), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(RT_device->sItr), &(RT_host->sItr), sizeof(ScreenItr), cudaMemcpyHostToDevice));

    //Copy scene static elements
    checkCudaErrors(cudaMemcpy(&(RT_device->scene.background), &(RT_host->scene.background), sizeof(Color), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(RT_device->scene.pSize), &(RT_host->scene.pSize), sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&(RT_device->scene.lSize), &(RT_host->scene.lSize), sizeof(int), cudaMemcpyHostToDevice));

    //Copy primitives
    //Spheres
    Sphere *device_spheres;
    checkCudaErrors(cudaMalloc(&device_spheres, sp * sizeof(Sphere)));
    checkCudaErrors(cudaMemcpy((device_spheres), (sp_ptr), sp * sizeof(Sphere),cudaMemcpyHostToDevice));
    
    //Cylinders
    Cylinder *device_cylinders; 
    checkCudaErrors(cudaMalloc(&device_cylinders, cy * sizeof(Cylinder)));
    checkCudaErrors(cudaMemcpy((device_cylinders), (cy_ptr), cy * sizeof(Cylinder),cudaMemcpyHostToDevice));

    //Polygons
    Polygon **polygons_list;
    Vector *vertices_temp;
    polygons_list = new Polygon*[po];
    for(int i=0; i<po; i++){
        checkCudaErrors(cudaMalloc(&(polygons_list[i]), sizeof(Polygon)));
        checkCudaErrors(cudaMemcpy(&(polygons_list[i]->nVertices), &(po_ptr[i]->nVertices), sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&(polygons_list[i]->properties), &(po_ptr[i]->properties), sizeof(Properties), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&(polygons_list[i]->N), &(po_ptr[i]->N), sizeof(Vector), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMalloc(&(vertices_temp), (po_ptr[i]->nVertices) * sizeof(Vector)));
        for (int j=0; j<(po_ptr[i]->nVertices); j++){
            checkCudaErrors(cudaMemcpy(&(vertices_temp[j]), &((po_ptr[i]->vertices)[j]), sizeof(Vector), cudaMemcpyHostToDevice));
        }
        checkCudaErrors(cudaMemcpy(&(polygons_list[i]->vertices), &(vertices_temp), sizeof(Vector*), cudaMemcpyHostToDevice));
    }
    Polygon **device_polygons_list;
    checkCudaErrors(cudaMalloc(&(device_polygons_list), po * sizeof(Polygon*)));
    checkCudaErrors(cudaMemcpy((device_polygons_list), (polygons_list), po * sizeof(Polygon*), cudaMemcpyHostToDevice));

    copy_primitives<<<1,1>>>(RT_device, device_spheres, device_cylinders, device_polygons_list, sp, cy, po);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ int readFile(std::string filename, RayTracer *RT, RayTracer *RT_device) {
    // Open file
    std::ifstream inputFilePrimitives(filename);
    if (!inputFilePrimitives.is_open()) {
        std::cerr<< "Could not open the file 1st- '"
                 << filename << "'" << std::endl; 
        return 0;
    }
    //Read number of primitives
    std::string tokPrimitives; 
    int sp = 0, cy = 0, po = 0, ls = 0;
    while(inputFilePrimitives>>tokPrimitives) {
        if(tokPrimitives.compare("s") == 0) 
            sp++;
        else if(tokPrimitives.compare("c") == 0){
            cy++;}
        else if(tokPrimitives.compare("p") == 0){
            po++;}
        else if(tokPrimitives.compare("l") == 0){
            ls++;
        }
    }


    // Close file
    inputFilePrimitives.close();

    // Open file
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr<< "Could not open the file 2nd- '"
                 << filename << "'" << std::endl; 
        return 0;
    }
    //Aux variables
    Scene scene; Screen screen; Observer observer; Properties p;
    std::string tok;

    //Reserve space in scene
    //Allocate primitives 
    scene.allocPrimitives(sp+cy+po);
    
    Sphere *sp_ptr;
    if (sp > 0){ 
        std::cout<<"There are "<< sp <<" spheres \n";
        sp_ptr = new Sphere[sp];
        for (int i = 0; i < sp; i++){
            scene.addPrimitive(sp_ptr+i);          
        }
    }

    Cylinder *cy_ptr;
    if (cy > 0){
        std::cout<<"There are "<< cy <<" cylinders \n";
        cy_ptr = new Cylinder[cy];
        for (int i = 0; i < cy; i++){
            scene.addPrimitive(cy_ptr+i);
        }
    }

    Polygon **po_ptr;
    if (po > 0){
        std::cout<<"There are "<< po <<" polygons \n";
        po_ptr = new Polygon*[po];
        // for (int i = 0; i < po; i++){
        //     scene.addPrimitive(po_ptr+i);
        // }
    } 

    Light *ls_ptr;
    if (ls > 0){
        std::cout<<"There are "<< ls <<" light sources \n";
        ls_ptr = new Light[ls];
        scene.lights = ls_ptr;
    }

    // Iterate file
    while (inputFile>>tok) {
        if(tok.compare("b") == 0)
            readBackground(inputFile, &scene);
        else if(tok.compare("resolution") == 0)
            readScreen(inputFile, &screen);
        else if(tok.compare("l") == 0){
            readLight(inputFile, &scene, ls_ptr); ls_ptr++;}
        else if(tok.compare("s") == 0){
            readSphere(inputFile, &scene, p, sp_ptr);sp_ptr++;}
        else if(tok.compare("c") == 0){
            readCylinder(inputFile, &scene, p, cy_ptr); cy_ptr++;}
        else if(tok.compare("p") == 0){
            readPolygon(inputFile, &scene, p, po_ptr); po_ptr++;}
        else if(tok.compare("f") == 0)
            p = readProperties(inputFile);
        else if(tok.compare("v") == 0)
            readObserver(inputFile, &observer);
    }
    sp_ptr-=sp;
    cy_ptr-=cy;
    po_ptr-=po;
    // Close file
    inputFile.close();
    // Save data in RayTracer object
    RT->observer = observer;
    RT->screen = screen;
    RT->scene = scene;
    RT->setScreenItr(); //Set Screen iterator info
    std::cout<<"FILE READ"<<std::endl;
    copyToDevice(RT, RT_device, sp_ptr, cy_ptr, po_ptr, sp, cy, po);
    std::cout<<"RAYTRACER COPIED"<<std::endl;
    return 1;

}

__host__ void readBackground(std::ifstream &inputFile, Scene *scene) {
    Color color; inputFile>>color.R>>color.G>>color.B;
    scene->background = color;
}

__host__ void readLight(std::ifstream &inputFile, Scene *scene, Light *ls_ptr) {
    std::string token, line; 
    getline(inputFile, line);
    //Line stream  
    std::vector<std::string> tokens;  
    std::istringstream iss(line);
    while (iss >> token) 
        tokens.push_back(token);
    //Copy data
    Vector point;
    point.x = std::stof(tokens[0]);
    point.y = std::stof(tokens[1]);
    point.z = std::stof(tokens[2]);
    Color color;
    if(tokens.size() == 6){
        color.R = std::stof(tokens[3]);
        color.G = std::stof(tokens[4]);
        color.B = std::stof(tokens[5]);
    }
    else {
        color.R = 1, color.G = 1, color.B = 1;
    }
    ls_ptr->color = color;
    ls_ptr->position = point;
    scene->lSize++;
}

__host__ void readCylinder(std::ifstream &inputFile, Scene *scene, Properties p, Cylinder *cy_ptr) {
    Vector ct, cb; float r;
    inputFile>>ct.x>>ct.y>>ct.z>>r;
    inputFile>>cb.x>>cb.y>>cb.z>>r;
    // scene->addPrimitive(new Cylinder(ct, cb, r, p)); 
    *cy_ptr = Cylinder(ct, cb, r, p);
}

__host__ void readPolygon(std::ifstream &inputFile, Scene *scene, Properties p, Polygon **po_ptr) {
    int nVertices; inputFile>>nVertices;
    Vector *vertices = new Vector[nVertices];
    for(int i=0; i<nVertices; i++)
        inputFile>>vertices[i].x>>vertices[i].y>>vertices[i].z;

    // scene->addPrimitive(new Polygon(nVertices, vertices, p));
    *po_ptr = new Polygon(nVertices, vertices, p);
    scene->addPrimitive(*po_ptr);
}

__host__ void readSphere(std::ifstream &inputFile, Scene *scene, Properties p, Sphere *sp_ptr) {
    Vector point; float r;
    inputFile>>point.x>>point.y>>point.z>>r;
    // scene->addPrimitive(new Sphere(point, r, p));
    *sp_ptr = Sphere (point, r, p);
}

__host__ void readObserver(std::ifstream &inputFile, Observer *observer) {
    std::string tok;
    //Read from
    Vector from; NEXT;
    inputFile>>from.x>>from.y>>from.z;
    observer->from = from;
    //Read at
    Vector lookAt; NEXT;
    inputFile>>lookAt.x>>lookAt.y>>lookAt.z;
    observer->lookAt = lookAt;
    //Read up
    Vector up; NEXT;
    inputFile>>up.x>>up.y>>up.z;
    observer->up = up;
    //Read angle
    NEXT;
    inputFile>>observer->angle;
}

__host__ void readScreen(std::ifstream &inputFile, Screen *s) {
    inputFile>>s->width>>s->height;
}

__host__ Properties readProperties(std::ifstream &inputFile) {
    Color color; 
    inputFile>>color.R>>color.G>>color.B;
    float kd, ks, shine, t, iof;
    inputFile>>kd>>ks>>shine>>t>>iof;
    Properties p(color, kd, ks, shine, t, iof);
    return p;
}


#endif
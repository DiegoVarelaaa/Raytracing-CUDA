#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
/*-----------------------------------LIBRARIES-------------------------------------*/

#include "sceneParser.h"
#include "Raytracer.h"
#include "cudaErrors.h"

struct distanceStruct{
    Primitive *object; 
    Ray ray;
};
struct shortestDistanceStruct{
    Primitive *object; 
    Ray ray;
};

__global__ void intersectionTestOp(int xResolution, int yResolution, int primitivesSize, distanceStruct *distance_struct_array, RayTracer *RT) { //Kernel de renderización
    int i = threadIdx.x + blockIdx.x * blockDim.x;  //Calculo del índice i
    int j = threadIdx.y + blockIdx.y * blockDim.y;  //Calculo del ínidce j
    int k = blockIdx.z * blockDim.z;
    if((i >= xResolution) || (j >= yResolution) || (k >= primitivesSize)) return;        //Si se sale de las dimensiones retorna
    
    Vector xPixel = (RT->sItr.pixelWidth)*i;
    Vector yPixel = (RT->sItr.pixelHeight)*j;
    Ray primaryRay(RT->observer.from, (RT->sItr.scanLine - yPixel) + xPixel);

    // Intersection test
    int operationCounter = k*xResolution*yResolution + j*xResolution + i;
    Primitive *obj;
    (RT->scene.primitives[k])->rayIntersection(&primaryRay, &obj);
    distance_struct_array[operationCounter].ray = primaryRay;
    distance_struct_array[operationCounter].object = obj;

    printf("i:%d j:%d k:%d OpCounter:%d   Distance: %f \n", i, j, k, operationCounter, primaryRay.distance);
    // int pixelCounter = j*xResolution + i;
    // image[pixelCounter] = RT->shading(primaryRay, object, 0)*(255);
}

__global__ void intersectionTestConditionCheck(int xResolution, int yResolution, int primitivesSize, distanceStruct *distance_struct_array, shortestDistanceStruct *shortest_distance_struct_array) { //Kernel de renderización
    int i = threadIdx.x + blockIdx.x * blockDim.x;  //Calculo del índice i
    int j = threadIdx.y + blockIdx.y * blockDim.y;  //Calculo del ínidce j
    if((i >= xResolution) || (j >= yResolution)) return;        //Si se sale de las dimensiones retorna

    int pixelCounter = j*xResolution + i;
    int numPixels = xResolution*yResolution;
    Primitive *closest_object;
    Ray closest_ray;
    for(int n = 0; n < primitivesSize; n++) { 
        // printf("i:%d j:%d Distance: %f \n", i , j , distance_struct_array[(n*xResolution*yResolution)+pixelCounter].ray.distance);
        if(distance_struct_array[n*numPixels+pixelCounter].ray.distance < closest_ray.distance){
            closest_ray = distance_struct_array[n*numPixels+pixelCounter].ray;
            closest_object = distance_struct_array[n*numPixels+pixelCounter].object;
        }
    }
    shortest_distance_struct_array[pixelCounter].ray = closest_ray;
    shortest_distance_struct_array[pixelCounter].object = closest_object;

    printf("i:%d j:%d pixelCounter:%d Distance: %f \n", i, j, pixelCounter,closest_ray.distance);
}


int main(int argc, char *argv[]) {
     // Read NFF scene
    std::string folder("Scenes/");
    std::string extension(".nff");
    std::string filename;
    if(argc > 1)
        filename = folder + argv[1] + extension;
    else
        filename = folder + "demo" + extension;

    std::cout<<"FILE: "<<filename<<std::endl;
    // Raytracing
    RayTracer RT_host;

    // Allocate memory in device
    RayTracer *RT_device;
    checkCudaErrors(cudaMalloc((void **) &RT_device, sizeof(RayTracer)));

    if (!readFile(filename, &RT_host, RT_device)) return 0;

    //RENDERIZADO
    int xResolution = RT_host.screen.width;     //Resolución en x
    int yResolution = RT_host.screen.height;    //Resolución en y
    int x_threads = 4;    //Numero de threads en x
    int y_threads = 4;    //Numero de threads en y
    int primitivesSize = RT_host.scene.pSize;

    dim3 blocks_intersection_op(xResolution/x_threads+1, yResolution/y_threads+1, primitivesSize);   //Cálculo del número de bloques de intersección
    dim3 threads_intersection_op(x_threads, y_threads);            //Definición del número de threads de intersección
    dim3 blocks_intersection_cond_chck(xResolution/x_threads+1, yResolution/y_threads+1);   //Cálculo del número de bloques de checkeo de condición
    dim3 threads_intersection_cond_chck(x_threads, y_threads);            //Definición del número de threads de checkeo de condición
    
    Color *image;       //  Creación del apuntador del espacio a reservar
    distanceStruct *distance_struct_array;// Estructura con el distance array, el rayo, y el objeto
    shortestDistanceStruct *shortest_distance_struct_array;// Estructura con el el rayo, y el objeto de la menor distancia

    
    checkCudaErrors(cudaMalloc((void **) &(image), xResolution*yResolution*sizeof(Color)));
    checkCudaErrors(cudaMalloc((void **) &(distance_struct_array), xResolution*yResolution*primitivesSize*sizeof(distanceStruct)));
    checkCudaErrors(cudaMalloc((void **) &(shortest_distance_struct_array), xResolution*yResolution*sizeof(shortestDistanceStruct)));
    // Call render function
    printf("-CUDA Version\n");

    clock_t start, stop;    //Variables temporales
    start = clock();        //Inicio del reloj

    intersectionTestOp<<<blocks_intersection_op, threads_intersection_op>>>(xResolution, yResolution, primitivesSize, distance_struct_array, RT_device);  //Se lanza el kernel de intersección
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("\n\n\n");

    intersectionTestConditionCheck<<<blocks_intersection_cond_chck, threads_intersection_cond_chck>>>(xResolution, yResolution, primitivesSize, distance_struct_array, shortest_distance_struct_array);  //Se lanza el kernel de checkeo de condición
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
   
    //Copy image from host to device
    // RT_host.image = new Color[xResolution*yResolution];
    // checkCudaErrors(cudaMemcpy ((RT_host.image), (image),  xResolution*yResolution*sizeof(Color), cudaMemcpyDeviceToHost));
    cudaDeviceReset();
    //Image to ppm
    // RT_host.imageToPPM();
    stop = clock();     //Se detiene el reloj y se calcula el tiempo de renderización
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // clean up
    //Limpieza de la memoria compartida
    // checkCudaErrors(cudaDeviceSynchronize());
    // free_world<<<1,1>>>(d_list,d_world,d_camera);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaFree(d_camera));
    // checkCudaErrors(cudaFree(d_world));
    // checkCudaErrors(cudaFree(d_list));
    // checkCudaErrors(cudaFree(d_rand_state));
    // checkCudaErrors(cudaFree(d_rand_state2));
    // checkCudaErrors(cudaFree(fb));

    std::cout << "PROCESS FINISHED" <<std::endl; 
    return 0;
}

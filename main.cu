#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
/*-----------------------------------LIBRARIES-------------------------------------*/

#include "sceneParser.h"
#include "Raytracer.h"
#include "cudaErrors.h"

__global__ void render(int xResolution, int yResolution, Color *image, RayTracer *RT) { //Kernel de renderización
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;  //Calculo del índice i
    int j = threadIdx.y + blockIdx.y * blockDim.y;  //Calculo del ínidce j
    if((i >= xResolution) || (j >= yResolution)) return;        //Si se sale de las dimensiones retorna

    Primitive *object;
    int pixelCounter = j*xResolution + i;
    Vector yPixel = (RT->sItr.pixelHeight)*j;
    Vector xPixel = (RT->sItr.pixelWidth)*i;
    Ray primaryRay(RT->observer.from, (RT->sItr.scanLine - yPixel) + xPixel);
    RT->intersectionTest(&primaryRay, &object);
    // printf("I FINISHED THE INTERSECTION TEST \n"); 
    image[pixelCounter] = RT->shading(primaryRay, object, 0)*(255);
    // printf("IM DONE IN THE KERNEL \n");
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
    int x_blocks = 8;    //Numero de bloques en x
    int y_blocks = 8;    //Numero de bloques en y

    dim3 blocks(xResolution/x_blocks+1, yResolution/y_blocks+1);   //Calculo del número de bloques
    dim3 threads(x_blocks, y_blocks);            //Calculo del número de threads
    
    Color *image;       //  Creación del apuntador del espacio a reservar
    checkCudaErrors(cudaMalloc((void **) &(image), xResolution*yResolution*sizeof(Color)));
    // //Call render function
    printf("-CUDA Version\n");

    clock_t start, stop;    //Variables temporales
    start = clock();        //Inicio del reloj
    render<<<blocks, threads>>>(xResolution, yResolution, image, RT_device);  //Se lanza el kernel de renderización

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
   
    //Copy image from host to device
    RT_host.image = new Color[xResolution*yResolution];
    checkCudaErrors(cudaMemcpy ((RT_host.image), (image),  xResolution*yResolution*sizeof(Color), cudaMemcpyDeviceToHost));
    cudaDeviceReset();
    //Image to ppm
    RT_host.imageToPPM();
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

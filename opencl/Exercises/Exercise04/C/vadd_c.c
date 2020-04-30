//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//
//------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif


extern double wtime();       // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id );


//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------

typedef struct HostData {
    float* h_a;
    float* h_b;
    float* h_c;
} HostData;

typedef struct DeviceContext {
    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel
    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;
} DeviceContext;

DeviceContext deviceContext;

void add_vectors(HostData hData) {
    int err;
    int count = LENGTH;
    size_t global;                  // global domain size
    unsigned int correct;           // number of correct results
    // Write a and b vectors into compute device memory
    err = clEnqueueWriteBuffer(deviceContext.commands, deviceContext.d_a, CL_TRUE, 0, sizeof(float) * count, hData.h_a, 0, NULL, NULL);
    checkError(err, "Copying h_a to device at d_a");

    err = clEnqueueWriteBuffer(deviceContext.commands, deviceContext.d_b, CL_TRUE, 0, sizeof(float) * count, hData.h_b, 0, NULL, NULL);
    checkError(err, "Copying h_b to device at d_b");

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(deviceContext.ko_vadd, 0, sizeof(cl_mem), &deviceContext.d_a);
    err |= clSetKernelArg(deviceContext.ko_vadd, 1, sizeof(cl_mem), &deviceContext.d_b);
    err |= clSetKernelArg(deviceContext.ko_vadd, 2, sizeof(cl_mem), &deviceContext.d_c);
    err |= clSetKernelArg(deviceContext.ko_vadd, 3, sizeof(unsigned int), &count);
    checkError(err, "Setting kernel arguments");

    double rtime = wtime();

    // Execute the kernel over the entire range of our 1d input data set
    // letting the OpenCL runtime choose the work-group size
    global = count;
    err = clEnqueueNDRangeKernel(deviceContext.commands, deviceContext.ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // Wait for the commands to complete before stopping the timer
    err = clFinish(deviceContext.commands);
    checkError(err, "Waiting for kernel to finish");

    rtime = wtime() - rtime;
    printf("\nThe kernel ran in %lf seconds\n",rtime);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer( deviceContext.commands, deviceContext.d_c, CL_TRUE, 0, sizeof(float) * count, hData.h_c, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array!\n%s\n", err_code(err));
        exit(1);
    }

    // Test the results
    correct = 0;
    float tmp;
    int i;
    for(i = 0; i < count; i++)
    {
        tmp = hData.h_a[i] + hData.h_b[i];     // assign element i of a+b to tmp
        tmp -= hData.h_c[i];             // compute deviation of expected and output result
        if(tmp*tmp < TOL*TOL)        // correct if square deviation is less than tolerance squared
            correct++;
        else {
            printf("tmp %f h_a %f h_b %f h_c %f \n",tmp, hData.h_a[i], hData.h_b[i], hData.h_c[i]);
        }
    }

    // summarise results
    printf("%d out of %d results were correct.\n", correct, count);
}

int main(int argc, char** argv)
{
    int          err;               // error code returned from OpenCL calls
    
    float*       h_a = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_b = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float*       h_c = (float*) calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device
    float*       h_d = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_e = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_f = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float*       h_g = (float*) calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device

    // Fill vectors a and b with random float values
    int i = 0;
    int count = LENGTH;
    for(i = 0; i < count; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
        h_e[i] = rand() / (float)RAND_MAX;
        h_g[i] = rand() / (float)RAND_MAX;
    }

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &deviceContext.device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (deviceContext.device_id == NULL)
        checkError(err, "Finding a device");

    err = output_device_info(deviceContext.device_id);
    checkError(err, "Printing device output");

    // Create a compute context
    deviceContext.context = clCreateContext(0, 1, &deviceContext.device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    deviceContext.commands = clCreateCommandQueue(deviceContext.context, deviceContext.device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    deviceContext.program = clCreateProgramWithSource(deviceContext.context, 1, (const char **) & KernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build the program
    err = clBuildProgram(deviceContext.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(deviceContext.program, deviceContext.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel from the program
    deviceContext.ko_vadd = clCreateKernel(deviceContext.program, "vadd", &err);
    checkError(err, "Creating kernel");

    // Create the input (a, b) and output (c) arrays in device memory
    deviceContext.d_a  = clCreateBuffer(deviceContext.context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_a");

    deviceContext.d_b  = clCreateBuffer(deviceContext.context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_b");

    deviceContext.d_c  = clCreateBuffer(deviceContext.context,  CL_MEM_READ_WRITE, sizeof(float) * count, NULL, &err);
    checkError(err, "Creating buffer d_c");

    // C = A + B
    HostData data = {
        h_a, h_b, h_c
    };

    add_vectors(data);
    
    // D = C + E
    data.h_a = h_c;
    data.h_b = h_e;
    data.h_c = h_d;

    add_vectors(data);

    // F = D + G
    data.h_a = h_d;
    data.h_b = h_g;
    data.h_c = h_f;

    add_vectors(data);

    // cleanup then shutdown
    clReleaseMemObject(deviceContext.d_a);
    clReleaseMemObject(deviceContext.d_b);
    clReleaseMemObject(deviceContext.d_c);
    clReleaseProgram(deviceContext.program);
    clReleaseKernel(deviceContext.ko_vadd);
    clReleaseCommandQueue(deviceContext.commands);
    clReleaseContext(deviceContext.context);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}


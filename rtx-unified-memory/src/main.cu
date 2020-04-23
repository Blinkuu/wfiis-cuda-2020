#include <stdio.h>
#include "page_faults/page_faults.h"
int main()
{
    test_page_fault_gpu_only();
    test_page_fault_cpu_only();
    test_page_fault_cpu_gpu();
    test_page_fault_gpu_cpu();
}


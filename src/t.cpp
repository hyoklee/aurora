#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Select the GPU (Aurora uses Intel Data Center GPU Max)
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    const int N = 1024;
    size_t size = N * sizeof(int);

    // 2. Allocate Host Memory (CPU)
    std::vector<int> host_data(N, 10); // Fill with 10s

    // 3. Allocate Device Memory (GPU)
    int* device_ptr = sycl::malloc_device<int>(N, q);

    if (device_ptr == nullptr) {
        std::cerr << "Failed to allocate GPU memory\n";
        return -1;
    }

    // 4. Copy CPU memory to GPU memory
    // This is an asynchronous operation
    sycl::event copy_event = q.memcpy(device_ptr, host_data.data(), size);

    // Wait for the copy to finish before proceeding
    copy_event.wait();

    std::cout << "Successfully copied " << size << " bytes to GPU.\n";

    // 5. Clean up
    sycl::free(device_ptr, q);

    return 0;
}

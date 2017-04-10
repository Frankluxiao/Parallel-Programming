#ifndef OPT_KERNEL
#define OPT_KERNEL


void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins, uint32_t* G_bins);

/* Include below the function headers of any other functions that you implement */

void* AllocDev(size_t size);

void CpyToDev(void* Device, void* Host, size_t size);

void CpyFromDev(void* Host, void* Device, size_t size);

void FreeDev(void* Device);


#endif

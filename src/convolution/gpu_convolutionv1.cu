__global__ void _conv2D(element *signal, element *result, element *mask, unsigned width, unsigned height, int ks)
{
    int radius = ks / 2;
	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;

    element value = 0;
    for (int i = 0; i < ks; ++i)
    {
	    for (int j = 0; j < ks; ++j)
        {
            int ll_x = gl_ix - radius + j;
            int ll_y = gl_iy - radius + i;
            if (!(ll_x < 0 || ll_x >= width
               || ll_y < 0 || ll_y >= height))    
               value += signal[ll_y * width + ll_x] * mask[i * ks + j];
        }
    }

	// Gets result 
    result[gl_iy * width + gl_ix] = value;
}

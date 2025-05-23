#include <stdio.h>
#include <stdlib.h>

#include <dspline_fitting.h>

int main(void) {

    double x[20] = {0.00, 0.55, 1.07, 1.65, 2.17, 2.66, 3.17, 3.73, 4.27, 4.79, 5.33, 5.85, 6.41, 6.90, 7.44, 7.98, 8.52, 9.00, 9.48, 10.00};
    double y[20] = {2.41, 1.70, 1.97, 1.68, 2.76, 1.93, 3.73, 2.73, 2.90, 4.52, 3.66, 1.47, 0.84, -0.94, 0.52, 0.13, 1.76, 4.02, 5.24, 9.45};

    dspline* result = ddspline(x, y, 20, 0);

    for (int i = 0; i < result->nn; i++) {
        printf("fx[%d] = %lf, fy[%d] = %lf\n", i, result->fx[i], i, result->fy[i]);
    }    

    return 0;
}

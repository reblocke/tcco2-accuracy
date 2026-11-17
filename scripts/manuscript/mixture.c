#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#include <math.h>
#include <stdlib.h>

// Sum the quadrature components at each requested disagreement within a chain batch.
int mixture(int n, int points, const double *mean, const double *inv_sd,
            const double *log_coeff, const double *offsets, double *out) {
    double *work = malloc(sizeof(double) * n);
    if (!work) return 1;
    for (int j = 0; j < points; j++) {
        for (int i = 0; i < n; i++) {
            double z = (offsets[j] - mean[i]) * inv_sd[i];
            work[i] = log_coeff[i] - 0.5 * z * z;
        }
#ifdef __APPLE__
        vvexp(work, work, &n);
        vDSP_sveD(work, 1, out + j, n);
#else
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += exp(work[i]);
        out[j] = sum;
#endif
    }
    free(work);
    return 0;
}

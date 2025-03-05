#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

typedef enum {
    IJK,
    IKJ,
    JIK,
    JKI,
    KIJ,
    KJI
} IjkId;

// ミリ秒単位の時間を返す関数 (clock_gettime()をomp_get_wtime()と同じ感覚で使えるように構成)
double getTime(void) {

#ifdef _OPENMP

    return omp_get_wtime();

#endif  // ifdef _OPENMP

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L

    struct timespec ts;

    timespec_get(&ts, TIME_UTC);

    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1000000000);

#endif  // if __STDC_VERSION__ >= 201112L
#endif  // ifdef __STDC_VERSION

#ifdef CLOCK_MONOTONIC

    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);

    return (double)t.tv_sec + ((double)t.tv_nsec / 1000000000);

#endif // ifdef CLOCK_MONOTONIC

    return (double)clock() / CLOCKS_PER_SEC;
}

double calcGFLOPS(int n, double seconds) {

    double FLOP = (double)2 * n * n * n;
    double G_seconds = (double)1000000000 * seconds;

    double GFLOPS = FLOP / G_seconds;

    return GFLOPS;
}

double* createMatrix(int N) {

    double* M = (double*)calloc(N * N, sizeof(double));

    if(M == NULL) {
        fprintf(stderr, "Failed to allocate memory of matrix.\n");
        exit(EXIT_FAILURE);
    }

    return M;
}

void clearMatrix(double* M) {
    free(M);
}

void fillZero(int N, double* M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            M[i * N + j] = 0;
        }
    }
}

bool isSame(int N, double* M1, double* M2) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (M1[i * N + j] != M2[i * N + j]) {
                return false;
            }
        }
    }

    return true;
}

void simpleMM(int N, double* A, double* B, double* C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void ijkMM(int N, double* A, double* B, double* C, IjkId ijk_id) {
    switch(ijk_id) {
        case IJK:
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
                }
            }
            break;
        case IKJ:
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < N; k++) {
                    for (int j = 0; j < N; j++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
                }
            }
            break;
        case JIK:
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    for (int k = 0; k < N; k++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
                }
            }
            break;
        case JKI:
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    for (int i = 0; i < N; i++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
                }
            }
            break;
        case KIJ:
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
                }
            }
            break;
        case KJI:
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < N; j++) {
                    for (int i = 0; i < N; i++) {
                        C[i * N + j] += A[i * N + k] * B[k * N + j];
                    }
                }
            }
            break;
    }
}

void blockingMM(int N, double* A, double* B, double* C, int bi, int bj, int bk) {
    for (int i = 0; i < N; i += bi) {
        for (int k = 0; k < N; k += bk) {
            for (int j = 0; j < N; j += bj) {
                for (int ii = i; ii < (i + bi) && ii < N; ii++) {
                    for (int kk = k; kk < (k + bk) && kk < N; kk++) {
                        for (int jj = j; jj < (j + bj) && jj < N; jj++) {
                            C[ii * N + jj] += A[ii * N + kk] * B[kk * N + jj];
                        }
                    }
                }
            }
        } 
    }
}

void unrollingMM(int N, double* A, double* B, double* C, int ui, int uj, int uk) {
    for(int i = 0; i < N; i += ui) {
        for (int k = 0; k < N; k += uk) {
            for (int j = 0; j < N; j += uj) {
                int ti, tj, tk;
                switch(ui) {
                    case 8:
                        ti = i + 7;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 7:
                        ti = i + 6;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 6:
                        ti = i + 5;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 5:
                        ti = i + 4;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 4:
                        ti = i + 3;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 3:
                        ti = i + 2;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 2:
                        ti = i + 1;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                    case 1:
                        ti = i;
                        switch(uj) {
                            case 8:
                                tj = j + 7;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 7:
                                tj = j + 6;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 6:
                                tj = j + 5;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 5:
                                tj = j + 4;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 4:
                                tj = j + 3;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 3:
                                tj = j + 2;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 2:
                                tj = j + 1;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                            case 1:
                                tj = j;
                                switch(uk) {
                                    case 8:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj] + A[ti * N + (k + 7)] * B[(k + 7) * N + tj];
                                        break;
                                    case 7:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj] + A[ti * N + (k + 6)] * B[(k + 6) * N + tj];
                                        break;
                                    case 6:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj] + A[ti * N + (k + 5)] * B[(k + 5) * N + tj];
                                        break;
                                    case 5:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj] + A[ti * N + (k + 4)] * B[(k + 4) * N + tj];
                                        break;
                                    case 4:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj] + A[ti * N + (k + 3)] * B[(k + 3) * N + tj];
                                        break;
                                    case 3:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj] + A[ti * N + (k + 2)] * B[(k + 2) * N + tj];
                                        break;
                                    case 2:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj] + A[ti * N + (k + 1)] * B[(k + 1) * N + tj];
                                        break;
                                    case 1:
                                        C[ti * N + tj] += A[ti * N + k] * B[k * N + tj];
                                        break;
                                }
                        }
                }
            }
        }
    }
}

#include <stdio.h>
#include "mm.h"

int main(void) {

    //
    // ユーザプログラムの準備
    //

    int N = 1000;

    double* A = createMatrix(N);
    double* B = createMatrix(N);
    double* C = createMatrix(N);
    double* ans = createMatrix(N);

    // 正解行列の生成
    simpleMM(N, A, B, ans);

    //
    // 主要ループ
    //

    for (int i = 0; i < 5; i++) {

        int block_i = 100;
        int block_j = 100;
        int block_k = 100;

        double s_time = getTime();

        // 評価対象部 (行列行列積)
        blockingMM(N, A, B, C, block_i, block_j, block_k);

        double e_time = getTime();

        // 計算速度出力
        double calc_time = e_time - s_time;
        printf("[%d:%d:%d] %lf [s], %lf [GFLOPS]\n", block_i, block_j, block_k, calc_time, calcGFLOPS(N, calc_time));

        if(!isSame(N, C, ans)) {
            printf("Error : incorrect result\n");
            exit(EXIT_FAILURE);
        }
    }

    //
    // 事後処理
    //

    clearMatrix(ans);
    clearMatrix(C);
    clearMatrix(B);
    clearMatrix(A);

    return 0;
}

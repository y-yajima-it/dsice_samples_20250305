#include <stdio.h>
#include "mm.h"

#include <dsice.h>

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

    dsice_tuner_id dsice_id = DSICE_CREATE();
    
    double options_i[5] = {1, 2, 4, 5, 8};
    double options_j[5] = {1, 2, 4, 5, 8};
    double options_k[5] = {1, 2, 4, 5, 8};

    DSICE_APPEND_PARAMETER(dsice_id, 5, options_i);
    DSICE_APPEND_PARAMETER(dsice_id, 5, options_j);
    DSICE_APPEND_PARAMETER(dsice_id, 5, options_k);

    //
    // 主要ループ
    //

    for (int i = 0; i < 50; i++) {

        double* tmp = DSICE_BEGIN(dsice_id);

        int unroll_i = tmp[0];
        int unroll_j = tmp[1];
        int unroll_k = tmp[2];

        double s_time = getTime();

        // 評価対象部 (行列行列積)
        unrollingMM(N, A, B, C, unroll_i, unroll_j, unroll_k);

        double e_time = getTime();

        // 計算速度出力
        double calc_time = e_time - s_time;
        printf("[%d:%d:%d] %lf [s], %lf [GFLOPS]\n", unroll_i, unroll_j, unroll_k, calc_time, calcGFLOPS(N, calc_time));

        if(!isSame(N, C, ans)) {
            printf("Error : incorrect result\n");
            exit(EXIT_FAILURE);
        }

        DSICE_END(dsice_id, calc_time);

        DSICE_PRINT_SIMPLE_LOOP_LOG_STD(dsice_id);
    }

    DSICE_PRINT_TUNING_RESULT_STD(dsice_id);

    DSICE_DELETE(dsice_id);

    //
    // 事後処理
    //

    clearMatrix(ans);
    clearMatrix(C);
    clearMatrix(B);
    clearMatrix(A);

    return 0;
}

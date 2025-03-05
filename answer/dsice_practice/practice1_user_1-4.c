#include <stdio.h>
#include "mm.h"

#include <dsice.h>

void printIjk(IjkId id) {
    switch(id) {
        case IJK:
            printf("[IJK] ");
            break;
        case IKJ:
            printf("[IKJ] ");
            break;
        case JIK:
            printf("[JIK] ");
            break;
        case JKI:
            printf("[JKI] ");
            break;
        case KIJ:
            printf("[KIJ] ");
            break;
        case KJI:
            printf("[KJI] ");
            break;
    }
}

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
    double options[6] = {IJK, IKJ, JIK, JKI, KIJ, KJI};
    DSICE_APPEND_PARAMETER(dsice_id, 6, options);

    //
    // 主要ループ
    //

    for (int i = 0; i < 10; i++) {

        double* tmp = DSICE_BEGIN(dsice_id);
        IjkId ijk_option = tmp[0];

        double s_time = getTime();

        // 評価対象部 (行列行列積)
        ijkMM(N, A, B, C, ijk_option);

        double e_time = getTime();

        // 計算速度出力
        double calc_time = e_time - s_time;
        printIjk(ijk_option);
        printf("%lf [s], %lf [GFLOPS]\n", calc_time, calcGFLOPS(N, calc_time));

        if(!isSame(N, C, ans)) {
            printf("Error : incorrect result\n");
            exit(EXIT_FAILURE);
        }

        DSICE_END(dsice_id, calc_time);
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "pascalops.h"

// Usando defines para os parâmetros de entrada
#define NX 512
#define NY 512
#define DT 0.001
#define NU 0.01

int main(int argc, char *argv[]) {
    // NOVO: Verificação do número de argumentos
    if (argc != 2) {
        return 1; // Retorna um código de erro
    }

    // Leitura dos parâmetros de entrada
    double NT = atof(argv[1]); // Número de passos de tempo

    // ALTERAÇÃO: Alocação de memória contígua
    // Aloca um único bloco para cada array e mapeia os ponteiros
    double **u = (double**)malloc(NX * sizeof(double*));
    double **v = (double**)malloc(NX * sizeof(double*));
    double **un = (double**)malloc(NX * sizeof(double*));
    double **vn = (double**)malloc(NX * sizeof(double*));
    
    double *u_data = (double*)malloc(NX * NY * sizeof(double));
    double *v_data = (double*)malloc(NX * NY * sizeof(double));
    double *un_data = (double*)malloc(NX * NY * sizeof(double));
    double *vn_data = (double*)malloc(NX * NY * sizeof(double));
    
    for (int i = 0; i < NX; i++) {
        u[i] = &u_data[i * NY];
        v[i] = &v_data[i * NY];
        un[i] = &un_data[i * NY];
        vn[i] = &vn_data[i * NY];
    }

    // Inicialização (paralela com collapse)
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double dx = i - NX/2;
            double dy = j - NY/2;
            double dist_sq = dx*dx + dy*dy;
            
            u[i][j] = 1.0;
            v[i][j] = 0.0;
            if (dist_sq < 400.0) {
                double perturbation = exp(-dist_sq / 100.0);
                u[i][j] += 2.0 * perturbation;
                v[i][j] += 1.5 * perturbation;
            }
        }
    }

    double start = omp_get_wtime();
    
    // ALTERAÇÃO: Região paralela única para todo o loop de tempo
    #pragma omp parallel
    {
        for (int t = 0; t < NT; t++) {
            // Evolução (agora apenas com "for", dentro da região paralela)
            #pragma omp for collapse(2) schedule(guided)
            for (int i = 1; i < NX-1; i++) {
                for (int j = 1; j < NY-1; j++) {
                    double laplacian_u = u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j];
                    double laplacian_v = v[i+1][j] + v[i-1][j] + v[i][j+1] + v[i][j-1] - 4*v[i][j];
                    
                    un[i][j] = u[i][j] + DT * NU * laplacian_u;
                    vn[i][j] = v[i][j] + DT * NU * laplacian_v;
                }
            }
            // Condições de contorno (agora apenas com "for")
            #pragma omp for schedule(guided)
            for (int i = 0; i < NX; i++) {
                un[i][0] = un[i][NY-2];
                un[i][NY-1] = un[i][1];
                vn[i][0] = vn[i][NY-2];
                vn[i][NY-1] = vn[i][1];
            }
            #pragma omp for schedule(guided)
            for (int j = 0; j < NY; j++) {
                un[0][j] = un[NX-2][j];
                un[NX-1][j] = un[1][j];
                vn[0][j] = vn[NX-2][j];
                vn[NX-1][j] = vn[1][j];
            }
            
            // ALTERAÇÃO: Barreira explícita para garantir que todos os cálculos terminaram antes do swap
            #pragma omp barrier

            // ALTERAÇÃO: Trocar ponteiros (operação serial, executada por apenas um thread)
            #pragma omp single
            {
                double **ut = u, **vt = v;
                u = un; v = vn;
                un = ut; vn = vt;
            }
        }
    }
    
    double end = omp_get_wtime();
    printf("%.6f\n", end - start);
    
    // Liberar memória (alterado para liberar apenas os blocos principais)
    free(u_data);
    free(v_data);
    free(un_data);
    free(vn_data);
    
    free(u);
    free(v);
    free(un);
    free(vn);
    
    return 0;
}

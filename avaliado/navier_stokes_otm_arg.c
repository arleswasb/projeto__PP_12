#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NX 512
#define NY 512
#define NT 2000
#define DT 0.001
#define NU 0.01

int main(int argc, char *argv[]) {
    // Verificação do número de argumentos
    if (argc != 5) {
        return 1; // Retorna um código de erro
    }

    // Leitura dos parâmetros de entrada
    double perturb_raio_sq = atof(argv[1]);
    double perturb_suavidade = atof(argv[2]);
    double perturb_amp_x = atof(argv[3]);
    double perturb_amp_y = atof(argv[4]);

    // Alocar memória
    double **u = malloc(NX * sizeof(double*));
    double **v = malloc(NX * sizeof(double*));
    double **un = malloc(NX * sizeof(double*));
    double **vn = malloc(NX * sizeof(double*));
    
    for (int i = 0; i < NX; i++) {
        u[i] = malloc(NY * sizeof(double));
        v[i] = malloc(NY * sizeof(double));
        un[i] = malloc(NY * sizeof(double));
        vn[i] = malloc(NY * sizeof(double));
    }
    
    // Inicialização paralela com collapse
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            double dx = i - NX/2;
            double dy = j - NY/2;
            double dist_sq = dx*dx + dy*dy;
            
            u[i][j] = 1.0;
            v[i][j] = 0.0;

            // CORREÇÃO AQUI: Usando as variáveis em minúsculas
            if (dist_sq < perturb_raio_sq) {
                double perturbation = exp(-dist_sq / perturb_suavidade);
                u[i][j] += perturb_amp_x * perturbation;
                v[i][j] += perturb_amp_y * perturbation;
            }
        }
    }
    
    double start = omp_get_wtime();
    
    // Loop principal
    for (int t = 0; t < NT; t++) {
        #pragma omp parallel for collapse(2) schedule(guided)
        for (int i = 1; i < NX-1; i++) {
            for (int j = 1; j < NY-1; j++) {
                double laplacian_u = u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4*u[i][j];
                double laplacian_v = v[i+1][j] + v[i-1][j] + v[i][j+1] + v[i][j-1] - 4*v[i][j];
                
                un[i][j] = u[i][j] + DT * NU * laplacian_u;
                vn[i][j] = v[i][j] + DT * NU * laplacian_v;
            }
        }
        
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < NX; i++) {
            un[i][0] = un[i][NY-2];
            un[i][NY-1] = un[i][1];
            vn[i][0] = vn[i][NY-2];
            vn[i][NY-1] = vn[i][1];
        }
        
        #pragma omp parallel for schedule(guided)
        for (int j = 0; j < NY; j++) {
            un[0][j] = un[NX-2][j];
            un[NX-1][j] = un[1][j];
            vn[0][j] = vn[NX-2][j];
            vn[NX-1][j] = vn[1][j];
        }
        
        // Trocar ponteiros
        double **ut = u, **vt = v;
        u = un; v = vn;
        un = ut; vn = vt;
    }
    
    double end = omp_get_wtime();
    printf("%.6f\n", end - start);
    
    // Liberar memória
    for (int i = 0; i < NX; i++) {
        free(u[i]); free(v[i]); free(un[i]); free(vn[i]);
    }
    free(u); free(v); free(un); free(vn);
    
    return 0;
}
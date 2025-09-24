inicializa ambiente virtual

source env.sh

comando para criar o executavel

gcc navier_stokes_otm_arg.c -fopenmp -lm -lmpascalops -o navier_stokes_otm_arg



comando para executar o pascalanalyzer


pascalanalyzer ./navier_stokes_otm_arg -t aut -c 1:32 -i 400.0,100.0,2.0,1.5 -o navier_simples.json

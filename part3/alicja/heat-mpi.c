/*
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage( char *s )
{
    fprintf(stderr, 
	    "Usage: %s <input file> [result file]\n\n", s);
}

int main( int argc, char *argv[] )
{
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;
    // TODO declare an array of request objects for non-blocking boundary exchange communication
    MPI_Request r[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
      printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs-1);

    // algorithmic parameters
    algoparam_t param;
    int np;

    double runtime, flop;
    double residual=0.0;
    // TODO declare a global residual
    double gresidual=0.0;

    // check arguments
    if( argc < 2 )
    {
	usage( argv[0] );
	return 1;
    }

    // check input file
    if( !(infile=fopen(argv[1], "r"))  ) 
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);
      
	usage(argv[0]);
	return 1;
    }

    // check result file
    resfilename= (argc>=3) ? argv[2]:"heat.ppm";

    if( !(resfile=fopen(resfilename, "w")) )
    {
	fprintf(stderr, 
		"\nError: Cannot open \"%s\" for writing.\n\n", 
		resfilename);
	usage(argv[0]);
	return 1;
    }

    // check input
    if( !read_input(infile, &param) )
    {
	fprintf(stderr, "\nError: Error parsing input file.\n\n");
	usage(argv[0]);
	return 1;
    }
    print_params(&param);

    // set the visualization resolution
    
    param.u     = 0;
    param.uhelp = 0;
    param.uvis  = 0;
    param.visres = param.resolution;
   
    if( !initialize(&param) )
	{
	    fprintf(stderr, "Error in Solver initialization.\n\n");
	    usage(argv[0]);
            return 1;
	}

    // TODO declare an array of request objects for non-blocking communication of final values of segments
    MPI_Request request[numprocs-1];

    // full size (param.resolution are only the inner points)
    np = param.resolution + 2;

    // TODO compute number of rows per worker
    int rows = param.resolution/numprocs;

    // starting time
    runtime = wtime();

    // send to workers the necessary data to perform computation
    for (int i=0; i<numprocs; i++) {
        if (i>0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // TODO send distinct segment (with ghost points) to each worker
                MPI_Send(&param.u[i*(rows*np)], (rows+2)*(np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[i*(rows*np)], (rows+2)*(np), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }

    iter = 0;
    while(1) {
	switch( param.algorithm ) {
	    case 0: // JACOBI
	            // TODO compute values for internal points of the first segment of the matrix
	            residual = relax_jacobi(param.u, param.uhelp, rows+2, np);

            // TODO send the last row to the next process (after the first iteration of computing values, because ghost points haven't changed yet)
            MPI_Send(&param.uhelp[rows*np], np, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD);

		    // Copy uhelp into u
		    // TODO update values of the first segment (with ghost points - what for, if they were only read?) of the matrix
		    for (int i=0; i<(rows+2); i++)
    		        for (int j=0; j<np; j++)
	    		    param.u[ i*np+j ] = param.uhelp[ i*np+j ];

	        // TODO receive the first row from the next process and save it in the last row (to update ghost points for the next iteration)
            MPI_Recv(&param.u[(rows*np)+np], np, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, &status);

		    break;
	    case 1: // RED-BLACK
		    residual = relax_redblack(param.u, np, np);
		    break;
	    case 2: // GAUSS
		    residual = relax_gauss(param.u, np, np);
		    break;
	    }

        iter++;

        // TODO get the current value of global residual
        MPI_Allreduce(&residual, &gresidual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // solution good enough ?
        if (gresidual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (param.maxiter>0 && iter>=param.maxiter) break;
    }

    // TODO receive final values of internal points of each segment from workers
    for (int i=1; i<numprocs; i++)
        MPI_Irecv(&param.u[i*(rows*np)+np], rows*np, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request[i-1]);

    // TODO wait until all workers send final values for their segments
    MPI_Waitall(numprocs-1, request, MPI_STATUS_IGNORE);

    // Flop count after iter iterations
    flop = iter * 11.0 * param.resolution * param.resolution;
    // stopping time
    runtime = wtime() - runtime;

    fprintf(stdout, "Time: %04.3f ", runtime);
    fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n", 
	    flop/1000000000.0,
	    flop/runtime/1000000);
    fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

    // for plot...
    coarsen( param.u, np, np,
	     param.uvis, param.visres+2, param.visres+2 );
  
    write_image( resfile, param.uvis,  
		 param.visres+2, 
		 param.visres+2 );

    finalize( &param );

    fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);

    MPI_Finalize();

    return 0;

} else {

    printf("I am worker %d and ready to receive work to do ...\n", myid);

    // receive information from master to perform computation locally

    int columns, rows, np;
    int iter, maxiter;
    int algorithm;
    double residual;
    // TODO declare variable that stores global residual
    double gresidual=0.0;

    MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    // TODO define number of rows in a segment computed by each worker
    rows = columns / numprocs;
    np = columns + 2;

    // TODO reduce memory allocated to each worker to fit only one segment with ghost points
    // allocate memory for worker
    double * u = calloc( sizeof(double),(rows+2)*(columns+2) );
    double * uhelp = calloc( sizeof(double),(rows+2)*(columns+2) );
    if( (!u) || (!uhelp) )
    {
        fprintf(stderr, "Error: Cannot allocate memory\n");
        return 0;
    }
    
    // fill initial values for matrix with values received from master
    // TODO receive a single distinct segment from master (with ghost points)
    MPI_Recv(&u[0], (rows+2)*(columns+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&uhelp[0], (rows+2)*(columns+2), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

    iter = 0;
    while(1) {
	switch( algorithm ) {
	    case 0: // JACOBI
	            // TODO compute values for assigned segment of the matrix
	            residual = relax_jacobi(u, uhelp, rows+2, np);

	        //  TODO send the last row (of internal points) to the next process, except the last process
	        if (myid < numprocs-1)
            MPI_Isend(&uhelp[rows*np], np, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, &r[2]);

            // TODO send the first row (of internal points) to the previous process
            MPI_Isend(&uhelp[np], np, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &r[0]);

		    // Copy uhelp into u
		    // TODO update values of computed segment of the matrix
		    for (int i=0; i<(rows+2); i++)
    		        for (int j=0; j<np; j++)
	    		    u[ i*np+j ] = uhelp[ i*np+j ];

		    // TODO receive the last row from the previous process and store it in the first row (after the first iteration of computing values, because ghost points haven't changed yet)
		    MPI_Irecv(&u[0], np, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &r[1]);

            if (myid < numprocs-1)
            // TODO receive the first row from the next process and store it in the last row
            MPI_Irecv(&u[(rows*np)+np], np, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, &r[3]);

            // TODO wait until the exchange of the first row is complete (r[0] and r[1])
            MPI_Waitall(2, r, MPI_STATUS_IGNORE);

            // TODO wait until the exchange of the last row is complete (r[2] and r[3])
            if (myid < numprocs-1)
                MPI_Waitall(2, &r[2], MPI_STATUS_IGNORE);

            break;
	    case 1: // RED-BLACK
		    residual = relax_redblack(u, np, np);
		    break;
	    case 2: // GAUSS
		    residual = relax_gauss(u, np, np);
		    break;
	    }

        iter++;

        // TODO get the current value of global residual
        MPI_Allreduce(&residual, &gresidual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // solution good enough ?
        if (gresidual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (maxiter>0 && iter>=maxiter) break;
    }
        // TODO send final values of internal points to master
        MPI_Send(&u[np], rows*np, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    if( u ) free(u); 
    if( uhelp ) free(uhelp);

    fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, residual);

    MPI_Finalize();
    exit(0);
  }
}

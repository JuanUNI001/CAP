#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "mpi.h"

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simulation parameters
static const unsigned int N = 500;

static const float SOURCE_TEMP = 5000.0f;
static const float BOUNDARY_TEMP = 1000.0f;

static const float MIN_DELTA = 0.05f;
static const unsigned int MAX_ITERATIONS = 20000;


static unsigned int idx(unsigned int x, unsigned int y, unsigned int stride) {
    return y * stride + x;
}


static void init(unsigned int source_x, unsigned int source_y, float * matrix) {
	// init
	memset(matrix, 0, N * N * sizeof(float));

	// place source
	matrix[idx(source_x, source_y, N)] = SOURCE_TEMP;

	// fill borders
	for (unsigned int x = 0; x < N; ++x) {
		matrix[idx(x, 0,   N)] = BOUNDARY_TEMP;
		matrix[idx(x, N-1, N)] = BOUNDARY_TEMP;
	}
	for (unsigned int y = 0; y < N; ++y) {
		matrix[idx(0,   y, N)] = BOUNDARY_TEMP;
		matrix[idx(N-1, y, N)] = BOUNDARY_TEMP;
	}
}


static void step(unsigned int source_x, unsigned int source_y, unsigned int local_n, unsigned int start_y, const float * current, float * next) {
	for (unsigned int y = 1; y <= local_n; ++y) {
		for (unsigned int x = 1; x < N-1; ++x) {
			if ((start_y + y - 1 == source_y) && (x == source_x)) {
				continue;
			}
			next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
			current[idx(x-1, y, N)] +
			current[idx(x+1, y, N)] +
			current[idx(x, y+1, N)]) / 4.0f;
		}
	}
}


static float diff(const float * current, const float * next, unsigned int local_n) {
	float maxdiff = 0.0f;
	for (unsigned int y = 1; y <= local_n; ++y) {
		for (unsigned int x = 1; x < N-1; ++x) {
			maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
		}
	}
	float global_maxdiff;
	MPI_Allreduce(&maxdiff, &global_maxdiff, 1, MPI_FLOAT,
	              MPI_MAX, MPI_COMM_WORLD);

	return global_maxdiff;
}

static void exchange_ghosts(float * current, unsigned int local_n,
                            int rank, int size) {

	int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
	int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

	// Send first owned row upwards and receive bottom ghost row.
	MPI_Sendrecv(&current[idx(0, 1, N)], N, MPI_FLOAT, up, 0,
	             &current[idx(0, local_n + 1, N)], N, MPI_FLOAT,
	             down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// Send last owned row downwards and receive top ghost row.
	MPI_Sendrecv(&current[idx(0, local_n, N)], N, MPI_FLOAT, down, 1,
	             &current[idx(0, 0, N)], N, MPI_FLOAT,
	             up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void write_png(float * current, int iter) {
	char file[100];
	uint8_t * image = malloc(3 * N * N * sizeof(uint8_t));
	float maxval = fmaxf(SOURCE_TEMP, BOUNDARY_TEMP);

	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 0; x < N; ++x) {
			unsigned int i = idx(x, y, N);
			colormap_rgb(COLORMAP_MAGMA, current[i], 0.0f, maxval, &image[3*i], &image[3*i + 1], &image[3*i + 2]);
		}
	}
	sprintf(file,"heat%i.png", iter);
	stbi_write_png(file, N, N, 3, image, 3 * N);

	free(image);
}


int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size > N-2) {
		if (rank == 0)
			printf("ERROR: Too many processes.\n");
		MPI_Finalize();
		return 1;
	}

	// --------------------------------------------------
	// Domain decomposition (1D horizontal strips)
	// --------------------------------------------------

	unsigned int rows = N-2;
	unsigned int local_n = rows / size;
	unsigned int remainder = rows % size;

	if (rank < remainder)
		local_n++;

	unsigned int start_y = 1 + rank * (rows / size)
	                         + (rank < remainder ? rank : remainder);

	// Local matrix includes two ghost rows.
	size_t array_size = (local_n + 2) * N * sizeof(float);

	float *current = malloc(array_size);
	float *next = malloc(array_size);

	// --------------------------------------------------
	// Initialize source and local boundary conditions
	// --------------------------------------------------

	unsigned int source_x = 0, source_y = 0;

	if (rank == 0) {
		srand(0);
		source_x = rand() % (N-2) + 1;
		source_y = rand() % (N-2) + 1;

		printf("Heat source at (%u, %u)\n", source_x, source_y);
	}

	MPI_Bcast(&source_x, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(&source_y, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	memset(current, 0, array_size);

	// Left and right boundaries.
	for (unsigned int y = 1; y <= local_n; ++y) {
		current[idx(0, y, N)] = BOUNDARY_TEMP;
		current[idx(N-1, y, N)] = BOUNDARY_TEMP;
	}

	// Top boundary (first process).
	if (rank == 0) {
		for (unsigned int x = 0; x < N; ++x)
			current[idx(x, 0, N)] = BOUNDARY_TEMP;
	}

	// Bottom boundary (last process).
	if (rank == size-1) {
		for (unsigned int x = 0; x < N; ++x)
			current[idx(x, local_n+1, N)] = BOUNDARY_TEMP;
	}

	// Heat source, if it belongs to this process.
	if (source_y >= start_y && source_y < start_y + local_n) {
		unsigned int local_y = source_y - start_y + 1;
		current[idx(source_x, local_y, N)] = SOURCE_TEMP;
	}

	memcpy(next, current, array_size);

	// --------------------------------------------------
	// Main simulation loop
	// --------------------------------------------------

	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	float t_diff = SOURCE_TEMP;
	unsigned int it;

	for (it = 0; it < MAX_ITERATIONS && t_diff > MIN_DELTA; ++it) {

		// Exchange ghost rows.
		exchange_ghosts(current, local_n, rank, size);

		// Compute local temperatures.
		step(source_x, source_y, local_n, start_y, current, next);

		// Calculate global maximum difference.
		t_diff = diff(current, next, local_n);

		if (rank == 0 && it % (MAX_ITERATIONS/10) == 0)
			printf("%u: %f\n", it, t_diff);

		// Swap matrices.
		float *swap = current;
		current = next;
		next = swap;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double stop = MPI_Wtime();

	double elapsed = stop - start;
	double max_elapsed;
	MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE,
	           MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0)
		printf("Computing time %f s.\n", max_elapsed);

	// --------------------------------------------------
	// Gather complete matrix on process 0
	// --------------------------------------------------

	int *recvcounts = malloc(size * sizeof(int));
	int *displs = malloc(size * sizeof(int));

	for (int p = 0; p < size; ++p) {
		unsigned int p_rows = rows / size + (p < remainder);
		unsigned int p_start = 1 + p * (rows / size)
		                         + (p < remainder ? p : remainder);

		recvcounts[p] = p_rows * N;
		displs[p] = p_start * N;
	}

	float *global = NULL;

	if (rank == 0) {
		global = malloc(N * N * sizeof(float));
		init(source_x, source_y, global);
	}

	MPI_Gatherv(&current[idx(0, 1, N)], local_n * N, MPI_FLOAT,
	            global, recvcounts, displs, MPI_FLOAT,
	            0, MPI_COMM_WORLD);

	if (rank == 0) {
		write_png(global, it);
		free(global);
	}

	free(recvcounts);
	free(displs);
	free(current);
	free(next);

	MPI_Finalize();

	return 0;
}

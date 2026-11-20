CC=gcc
MPICC=mpicc

CFLAGS=-O2 -fopenmp
LDFLAGS=-lm

NP=8

all: heat heat_parallel

heat: heat.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

heat_parallel: heat_parallel.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: all
	@echo "===== SERIAL VERSION ====="
	@/usr/bin/time -f "Elapsed time: %e seconds" ./heat

	@echo ""
	@echo "===== PARALLEL VERSION ($(NP) processes) ====="
	@/usr/bin/time -f "Elapsed time: %e seconds" mpirun -np $(NP) ./heat_parallel

clean:
	rm -f heat heat_parallel heat*.png

.PHONY: all run clean
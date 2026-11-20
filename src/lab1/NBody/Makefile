NAME=nbody

CC=icx

CFLAGS=-g -O3 -I. -xHost -qopenmp -qopt-report=3

DEPS = gtime.h memory.h nbody.h nbody_routines.h

OBJ = gtime.o memory.o nbody.o nbody_routines.o

LIB = -lm

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $< 

$(NAME): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIB)


clean: 
	rm -f *.o *.optrpt $(NAME) *.opt.yaml

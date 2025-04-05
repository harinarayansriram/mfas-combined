CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp
LDFLAGS = -lm -fopenmp

SOURCES = program.c solution_instance.c
OBJECTS = $(SOURCES:.c=.o)
EXECUTABLE = feed_forward

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
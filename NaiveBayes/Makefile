NAME_OF_EXECUTABLE_FILE = NaiveBayes

COMPILER = g++
COMPILER_OPTS = -c -g


LINKER = g++
LINKER_OPTS = -o $(NAME_OF_EXECUTABLE_FILE)


OBJS = main.o NBC.o


#  linking the program.

$(NAME_OF_EXECUTABLE_FILE) : $(OBJS)
	$(LINKER) $(LINKER_OPTS) $(OBJS)

# Rules for performing the compilation of each individual object file.


NBC.o : NBC.h NBC.cpp
	$(COMPILER) $(COMPILER_OPTS) NBC.cpp

main.o : main.cpp NBC.o
	$(COMPILER) $(COMPILER_OPTS) main.cpp

clean:
	-rm *.o $(NAME_OF_EXECUTABLE_FILE)



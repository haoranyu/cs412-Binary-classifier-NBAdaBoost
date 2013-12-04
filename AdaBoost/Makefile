NAME_OF_EXECUTABLE_FILE = NBAdaBoost

COMPILER = g++
COMPILER_OPTS = -c -g


LINKER = g++
LINKER_OPTS = -o $(NAME_OF_EXECUTABLE_FILE)


OBJS = main.o AB.o


#  linking the program.

$(NAME_OF_EXECUTABLE_FILE) : $(OBJS)
	$(LINKER) $(LINKER_OPTS) $(OBJS)

# Rules for performing the compilation of each individual object file.


AB.o : AB.h AB.cpp
	$(COMPILER) $(COMPILER_OPTS) AB.cpp

main.o : main.cpp AB.o
	$(COMPILER) $(COMPILER_OPTS) main.cpp

clean:
	-rm *.o $(NAME_OF_EXECUTABLE_FILE)



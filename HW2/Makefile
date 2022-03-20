CXX = g++
FLAG = -Wall
EXEC1 = naive_bayes_classifier 
EXEC2 = online_learning
INPUT = args
OUTPUT = out.txt


all: $(EXEC1) $(EXEC2)

%: %.cpp 
	$(CXX) $(FLAG) -o $@ $<

run_1: $(EXEC1)
	./$(EXEC1) < args > out.txt

run_2: $(EXEC2)
	./$(EXEC2) < args > out.txt

clean:
	rm *.o $(EXEC1) $(EXEC2)
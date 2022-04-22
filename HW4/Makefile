CXX = g++
FLAG = -Wall
EXEC1 = EM
INPUT = args
OUTPUT = out.txt


all: $(EXEC1)

%: %.cpp 
	$(CXX) $(FLAG) -o $@ $<

clean:
	rm *.o $(EXEC1)
CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -march=native -fopenmp
TARGET = minstd_rand
SOURCES = minstd_rand.cpp
VEC_REPORT = vectorization_report.txt

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

vec-report:
	$(CXX) $(CXXFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c $(SOURCES) 2> $(VEC_REPORT)

clean:
	rm -f $(TARGET) *.o $(VEC_REPORT)

.PHONY: all clean vec-report

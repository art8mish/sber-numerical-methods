CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -march=native
TARGET = matrix
SOURCES = matrix.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) $(BENCH_LOOP)

.PHONY: all clean bench-order

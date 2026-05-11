CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++20 -march=native
TARGET = perftest
SOURCES = perftest.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

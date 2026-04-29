CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -march=native
TARGET = minstd_rand
SOURCES = minstd_rand.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean

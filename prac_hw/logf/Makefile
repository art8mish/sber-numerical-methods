
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
TARGET = logf

SOURCES = logf.cpp


all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean
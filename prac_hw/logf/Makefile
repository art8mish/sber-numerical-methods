
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20
TARGET = logf.o

SOURCES = logf.cpp

# Тест: ULP vs std::log(double), errno/FE как у libm (без встроенной оптимизации log).
TEST_SRC = test_logf.cpp
TEST_BIN = test_logf
TESTFLAGS = -fno-builtin-log -fno-builtin-logf


all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -c -o $(TARGET) $(SOURCES)

$(TEST_BIN): $(TARGET) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $(TEST_BIN) $(TEST_SRC) $(TARGET) -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

check: test

clean:
	rm -f $(TARGET) $(TEST_BIN)

.PHONY: all clean check test

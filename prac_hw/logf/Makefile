
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O2
AVX2FLAGS = -mavx2 -mfma

TARGET = logf.o
SOURCES = logf.cpp

AVX2_OBJ = logf_avx2.o
AVX2_SRC = logf_avx2.cpp

TEST_SRC = test_logf.cpp
TEST_BIN = test_logf

TEST_AVX2_SRC = test_logf_avx2.cpp
TEST_AVX2_BIN = test_logf_avx2

TESTFLAGS = -fno-builtin-log -fno-builtin-logf


all: $(TARGET) $(AVX2_OBJ)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -c -o $(TARGET) $(SOURCES)

$(AVX2_OBJ): $(AVX2_SRC)
	$(CXX) $(CXXFLAGS) $(AVX2FLAGS) -c -o $(AVX2_OBJ) $(AVX2_SRC)

$(TEST_BIN): $(TARGET) $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $(TEST_BIN) $(TEST_SRC) $(TARGET) -lm

$(TEST_AVX2_BIN): $(AVX2_OBJ) $(TEST_AVX2_SRC)
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $(TEST_AVX2_BIN) $(TEST_AVX2_SRC) $(AVX2_OBJ) -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

test_avx2: $(TEST_AVX2_BIN)
	./$(TEST_AVX2_BIN)

check: test test_avx2

clean:
	rm -f $(TARGET) $(AVX2_OBJ) $(TEST_BIN) $(TEST_AVX2_BIN)

.PHONY: all clean check test test_avx2

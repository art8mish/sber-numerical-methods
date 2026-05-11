CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O2
PERFFLAGS = -O3 -march=native -Wall -Wextra -std=c++20
TESTFLAGS = -fno-builtin-log -fno-builtin-logf
LDLIBS = -lm

logf_avx2.o: CXXFLAGS += -mavx2 -mfma

all: logf.o logf_avx2.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test_logf: test_logf.cpp logf.o
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $@ $^ $(LDLIBS)

test_logf_avx2: test_logf_avx2.cpp logf_avx2.o
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $@ $^ $(LDLIBS)

perftest: perftest.cpp logf.o logf_avx2.o
	$(CXX) $(PERFFLAGS) $(TESTFLAGS) -o $@ $^ $(LDLIBS) -lpthread -ldl

test: test_logf
	./$<

test_avx2: test_logf_avx2
	./$<

perf: perftest
	./$<

check: test test_avx2

clean:
	rm -f logf.o logf_avx2.o test_logf test_logf_avx2 perftest

.PHONY: all clean check test test_avx2 perf

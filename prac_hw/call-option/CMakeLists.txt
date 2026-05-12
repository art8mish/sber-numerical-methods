cmake_minimum_required(VERSION 3.16)
project(call_options_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(call_options_mc call_options_mc.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(call_options_mc PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(call_options_mc PRIVATE /O2)
else()
    target_compile_options(
        call_options_mc
        PRIVATE
            -O3
            -ffp-contract=fast
    )
endif()

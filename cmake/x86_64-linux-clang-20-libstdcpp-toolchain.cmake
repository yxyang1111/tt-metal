set(CMAKE_SYSTEM_PROCESSOR "x86_64")

set(CMAKE_C_COMPILER clang-20 CACHE INTERNAL "C compiler")

set(CMAKE_CXX_COMPILER clang++-20 CACHE INTERNAL "C++ compiler")

# Use for configure time
set(ENABLE_LIBCXX FALSE CACHE INTERNAL "Using clang's libc++")

# Our build is super slow; put a band-aid on it by choosing a linker that can cope better.
# Old mold versions cannot link ThinLTO bitcode and will fall back to whatever ld.lld is on PATH.
# With clang-20 that can pick an older lld and fail to read LLVM 20 bitcode, so gate mold by version.
find_program(MOLD ld.mold)
if(MOLD)
    execute_process(
        COMMAND
            ${MOLD} --version
        OUTPUT_VARIABLE mold_version_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(mold_version_output MATCHES "mold ([0-9.]+)" AND CMAKE_MATCH_1 VERSION_GREATER_EQUAL "1.6")
        set(CMAKE_LINKER_TYPE MOLD)
    endif()
endif()

if(NOT DEFINED CMAKE_LINKER_TYPE)
    find_program(LLD ld.lld-20)
    if(LLD)
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()

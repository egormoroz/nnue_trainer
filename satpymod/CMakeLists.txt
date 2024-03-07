cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

project(satpymod)

add_executable(satpymod
    main.cpp zobrist.cpp pack.cpp pymod.cpp
    dataloader/batch.cpp dataloader/halfkp.cpp dataloader/batchstream.cpp
    board/board.cpp board/board_moves.cpp board/parse.cpp 
    board/validate.cpp board/see.cpp movgen/attack.cpp 
    movgen/magic.cpp movgen/generate.cpp primitives/utility.cpp
    nnue/evaluate.cpp core/eval.cpp)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(saturn PRIVATE Threads::Threads)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native")
endif()


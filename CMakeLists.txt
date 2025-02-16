cmake_minimum_required(VERSION 3.29)

# Set vcpkg toolchain file BEFORE the project() command
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(treehacks LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(websocketpp CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS asio system thread date_time regex serialization filesystem program_options random container)
find_package(OpenSSL REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)



# Add source files (all files in src/ plus main.cpp)
file(GLOB_RECURSE SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

add_executable(treehacks ${SOURCES}
        include/output_websocket.h
        src/output_websocket.cpp
        include/monte_carlo_simulation.h
        src/monte_carlo_simulation.cu
        src/simulation_update_thread.cpp
        src/data_updater.cpp
        include/data_updater.h
        include/ml_algorithm.h
        src/ml_algorithm.cpp
        include/svm_model.h
        src/svm_model.cpp
)

# Include directories
target_include_directories(treehacks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(treehacks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(treehacks PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(treehacks PRIVATE ${OPENSSL_INCLUDE_DIR})
target_include_directories(treehacks PRIVATE ${Python3_INCLUDE_DIRS})

# Link libraries
target_link_libraries(treehacks PRIVATE
        CUDA::cublas
        CUDA::cusolver
        websocketpp::websocketpp
        Boost::asio
        Boost::system
        Boost::thread
        Boost::date_time
        Boost::regex
        Boost::serialization
        Boost::filesystem
        Boost::program_options
        Boost::random
        Boost::container
        OpenSSL::SSL
        OpenSSL::Crypto
        ${Python3_LIBRARIES}

)

# Enable CUDA separable compilation
set_target_properties(treehacks PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

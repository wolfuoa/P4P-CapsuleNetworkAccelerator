cmake_minimum_required(VERSION 3.13)

project(capsNetTest)

set(CMAKE_CXX_STANDARD 11)

include_directories(
    ${CMAKE_SOURCE_DIR}/floating
    ${CMAKE_SOURCE_DIR}/util
)

add_executable(capsNetTest 
    CapsuleNetworkTestbench.cpp
    floating/CapsuleNetwork.cpp
    floating/DigitCaps.cpp
    floating/PrimaryCaps.cpp
    floating/ReLUConv1.cpp
    floating/CapsuleLayer.cpp
)
cmake_minimum_required( VERSION 3.0 )

project( work1 )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11" )

include_directories( "/usr/include/eigen3" )
add_executable( test test.cpp )
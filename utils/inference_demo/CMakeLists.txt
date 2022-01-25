cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# Import torch and MonopodSDK
find_package(Torch REQUIRED)
find_package(MonopodSdk REQUIRED)

add_executable(demo-inference demo_inference.cpp)
target_link_libraries(demo-inference PUBLIC
    "${TORCH_LIBRARIES}" 
    MonopodSdk::MonopodSdk
)
set_property(TARGET demo-inference PROPERTY CXX_STANDARD 14)
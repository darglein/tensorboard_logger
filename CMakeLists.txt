cmake_minimum_required(VERSION 3.16.3)
project(tensorboard_logger)

option(BUILD_TEST "Build test" OFF)


# -----------------------------------------------------------------------------
# Building the tensorboard_logger library
# -----------------------------------------------------------------------------

file(GLOB protos "proto/*.proto")


if (TARGET libprotobuf)
    # Create lists to hold the generated files
    set(PROTO_SRCS)
    set(PROTO_HDRS)

    # Loop over each .proto file and define how to generate the C++ code
    foreach (proto_file ${protos})
        # Figure out the output paths
        get_filename_component(proto_name ${proto_file} NAME_WE)
        get_filename_component(proto_dir ${proto_file} DIRECTORY)

        # Define the expected output files
        set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.cc")
        set(h_file "${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.h")

        # Add the custom command to run protoc
        add_custom_command(
                OUTPUT ${cpp_file} ${h_file}
                COMMAND protoc
                --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                --proto_path=${proto_dir}
                ${proto_file}
                DEPENDS ${proto_file} protoc # Depends on the file AND the compiler tool
                COMMENT "Generating C++ code for ${proto_name}.proto"
                VERBATIM
        )

        # Append to the list of sources
        list(APPEND PROTO_SRCS ${cpp_file})
        list(APPEND PROTO_HDRS ${h_file})
    endforeach ()

else ()

    find_package(Protobuf REQUIRED)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${protos})
endif ()

add_library(tensorboard_logger
        "src/crc.cc"
        "src/tensorboard_logger.cc"
        ${PROTO_SRCS}
)

target_compile_features(tensorboard_logger PRIVATE cxx_std_11)
target_compile_options(tensorboard_logger PRIVATE -Wall -O2)

target_include_directories(tensorboard_logger
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROTOBUF_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if (TARGET libprotobuf)
    target_link_libraries(tensorboard_logger PUBLIC libprotobuf)
else ()
    target_link_libraries(tensorboard_logger PUBLIC ${Protobuf_LIBRARIES})
endif ()


if (BUILD_TEST)
    add_executable(tensorboard_logger_test tests/test_tensorboard_logger.cc)
    target_compile_features(tensorboard_logger_test PRIVATE cxx_std_11)
    target_compile_options(tensorboard_logger_test PRIVATE -Wall -O2)
    target_include_directories(tensorboard_logger_test
            PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )
    target_link_libraries(tensorboard_logger_test tensorboard_logger)
endif ()

# -----------------------------------------------------------------------------
# Installing the tensorboard_logger library
# -----------------------------------------------------------------------------

install(
        TARGETS tensorboard_logger
        EXPORT tensorboard_logger_Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/tensorboard_loggerConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/tensorboard_loggerConfig.cmake"
        INSTALL_DESTINATION
        ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
        EXPORT tensorboard_logger_Targets
        FILE tensorboard_loggerTargets.cmake
        NAMESPACE tensorboard_logger::
        DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
        FILES "${PROJECT_BINARY_DIR}/tensorboard_loggerConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
        DIRECTORY include/
        DESTINATION include
)

install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.pb.h"
        PATTERN "CMakeFiles" EXCLUDE
)

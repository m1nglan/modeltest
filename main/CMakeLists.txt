# ESP-DL utilities for embedding binary data
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# JPG images to embed as rodata binary data
set(embed_jpg_files
    "t1.jpg"
    "t2.jpg"
)

idf_component_register(SRCS "main.cpp" "driver/classifier_driver.cpp"
                    INCLUDE_DIRS "." "driver"
                    REQUIRES espressif__esp-dl)

# Embed JPG files into rodata (creates _binary_t1_jpg_start etc.)
foreach(jpg_file ${embed_jpg_files})
    target_add_aligned_binary_data(${COMPONENT_LIB} "${CMAKE_CURRENT_SOURCE_DIR}/${jpg_file}" BINARY)
endforeach()

# Flash model.espdl to the "model" partition
esptool_py_flash_to_partition(flash "model" "${CMAKE_CURRENT_SOURCE_DIR}/model.espdl")

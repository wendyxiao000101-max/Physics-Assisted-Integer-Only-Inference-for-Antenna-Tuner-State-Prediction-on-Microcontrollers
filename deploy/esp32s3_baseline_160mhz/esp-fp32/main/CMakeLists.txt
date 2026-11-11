idf_component_register(
    SRCS "main.c" "fp32_runtime.c"
    INCLUDE_DIRS "."
    EMBED_FILES "fp32_model.bin"
    REQUIRES esp_hw_support freertos esp_partition
)

target_link_libraries(${COMPONENT_LIB} PRIVATE m)

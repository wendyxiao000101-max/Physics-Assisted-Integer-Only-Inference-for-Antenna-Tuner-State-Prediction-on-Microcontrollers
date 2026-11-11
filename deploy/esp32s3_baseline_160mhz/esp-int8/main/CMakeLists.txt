idf_component_register(
    SRCS "main.c" "integer_runtime.c" "model_blob.c"
    INCLUDE_DIRS "."
    REQUIRES esp_hw_support freertos esp_partition
)

cmake_minimum_required(VERSION 3.20)
project(rf-detr-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/ggml)

add_library(rfdetr STATIC
    src/ops.cpp
    src/backbone.cpp
    src/projector.cpp
    src/deform_attn.cpp
    src/decoder.cpp
    src/segmentation.cpp
    src/keypoints.cpp
    src/loss.cpp
    src/dataset.cpp
    src/rfdetr_capi.cpp
)
target_link_libraries(rfdetr PUBLIC ggml)
target_include_directories(rfdetr PUBLIC src include)
target_compile_definitions(rfdetr PRIVATE RFDETR_BUILD)

foreach(t test_backbone_smoke test_backbone test_backbone_base test_backbone_multi_image test_projector test_projector_multiscale test_deform_attn test_deform_attn_multilevel test_deform_attn_backward test_decoder test_decoder_base test_decoder_small test_decoder_medium test_decoder_large test_decoder_large_deprecated test_segmentation test_segmentation_small test_segmentation_medium test_segmentation_large test_segmentation_xlarge test_segmentation_preview test_keypoints test_norm_backward test_loss)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE rfdetr)
endforeach()

add_executable(seg_cli demos/seg_cli.cpp)
target_link_libraries(seg_cli PRIVATE rfdetr)

add_executable(train_step_demo demos/train_step_demo.cpp)
target_link_libraries(train_step_demo PRIVATE rfdetr)

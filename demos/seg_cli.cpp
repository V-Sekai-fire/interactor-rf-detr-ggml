// SPDX-License-Identifier: MIT
// seg_cli: RFDETRSegNano instance segmentation on real images, through the C++ internals.
//
// The C ABI (src/rfdetr_capi.cpp) is still a stub and returns boxes only, so this drives the
// same graph test_segmentation.cpp validates -- backbone, projector, decoder, segmentation
// head -- with the decoder's own in-graph top-k (no reference override).
//
// Input: pre-processed images, one file each: 312*312*3 float32, planar (C, H, W), already
// resized (bilinear) and ImageNet-normalised exactly as gen_reference_segmentation.py does.
// Output per input: <input>.seg, three float32 arrays back to back:
//   boxes  (num_queries, 4)          cx, cy, w, h in [0, 1]
//   logits (num_queries, 91)         class logits (COCO ids; person = 1)
//   masks  (num_queries, 78, 78)     mask logits, row-major per query
//
// usage: seg_cli MODEL_DIR IMAGE.f32 [IMAGE.f32 ...]
#include "backbone.h"
#include "decoder.h"
#include "projector.h"
#include "segmentation.h"
#include "../tests/test_common.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: seg_cli MODEL_DIR IMAGE.f32 [IMAGE.f32 ...]\n");
        return 2;
    }
    const std::string dir = argv[1];
    Model m;
    for (const char * part : { "backbone", "projector", "decoder", "segmentation" }) {
        std::string p = dir + "/rf-detr-seg-nano-" + part + ".gguf";
        if (!rfdetr_load(m, p.c_str())) {
            fprintf(stderr, "failed to load %s\n", p.c_str());
            return 1;
        }
    }

    BackboneParams bp;
    bp.hidden = 384; bp.n_layer = 12; bp.n_head = 6; bp.patch_size = 12; bp.n_register = 0; bp.num_windows = 1;
    bp.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
    bp.out_feature_indexes  = { 2, 5, 8, 11 };
    DecoderParams dp;
    dp.hidden_dim = 256; dp.dec_layers = 4; dp.sa_nheads = 8; dp.ca_nheads = 16; dp.dec_n_points = 2;
    dp.num_queries = 100; dp.num_classes = 91; dp.gw = 26; dp.gh = 26;
    SegmentationParams sp;
    sp.hidden_dim = 256; sp.num_blocks = 4; sp.downsample_ratio = 4; sp.image_w = 312; sp.image_h = 312;
    const int64_t res = 312;
    const size_t n_px = (size_t) res * res * 3;

    const size_t max_nodes = 200000;
    init_graph_ctx(m, max_nodes);
    ggml_tensor * x = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, res, res, 3, 1);
    ggml_set_name(x, "pixel_values");
    ggml_set_input(x);
    std::vector<ggml_tensor *> taps = dinov2_backbone(m, x, bp);
    ggml_tensor * fused = projector_p4(m, taps, 256);
    ggml_tensor * memory = ggml_reshape_3d(m.ctx_g, fused, dp.gw * dp.gh, dp.hidden_dim, 1);
    memory = ggml_cont(m.ctx_g, ggml_permute(m.ctx_g, memory, 1, 0, 2, 3));
    DecoderOutput dout = rfdetr_decoder(m, memory, dp);   // in-graph top-k
    ggml_tensor * mask = segmentation_head(m, fused, dout.hidden_states, sp).back();   // (78, 78, 100, 1)

    ggml_backend_t backend = rfdetr_backend_init();
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, 16);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    for (ggml_tensor * t : { mask, dout.pred_boxes, dout.pred_logits }) {
        ggml_build_forward_expand(gf, t);
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        return 1;
    }
    const std::vector<float> proposals = output_proposals_data(dp.gw, dp.gh);

    std::vector<float> pixels(n_px);
    for (int i = 2; i < argc; i++) {
        FILE * f = fopen(argv[i], "rb");
        if (!f) {
            fprintf(stderr, "%s: cannot open\n", argv[i]);
            continue;
        }
        size_t got = fread(pixels.data(), sizeof(float), n_px, f);
        fclose(f);
        if (got != n_px) {
            fprintf(stderr, "%s: expected %zu floats, got %zu\n", argv[i], n_px, got);
            continue;
        }
        ggml_backend_tensor_set(x, pixels.data(), 0, n_px * sizeof(float));
        ggml_backend_tensor_set(dout.output_proposals, proposals.data(), 0, proposals.size() * sizeof(float));
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", argv[i]);
            continue;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::string out = std::string(argv[i]) + ".seg";
        FILE * o = fopen(out.c_str(), "wb");
        if (!o) {
            fprintf(stderr, "%s: cannot write\n", out.c_str());
            continue;
        }
        for (ggml_tensor * t : { dout.pred_boxes, dout.pred_logits, mask }) {
            std::vector<float> buf(ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            fwrite(buf.data(), sizeof(float), buf.size(), o);
        }
        fclose(o);
        printf("%s %.1f ms\n", out.c_str(), ms);
        fflush(stdout);
    }
    return 0;
}

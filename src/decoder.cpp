#include "decoder.h"
#include "deform_attn.h"
#include "keypoints.h"

#include <cmath>

std::vector<float> output_proposals_data(int gw, int gh) {
    std::vector<float> out((size_t) gw * gh * 4);
    for (int row = 0; row < gh; row++) {
        for (int col = 0; col < gw; col++) {
            float * p = &out[((size_t) row * gw + col) * 4];
            p[0] = (col + 0.5f) / gw; // cx
            p[1] = (row + 0.5f) / gh; // cy
            p[2] = 0.05f;             // w
            p[3] = 0.05f;             // h
        }
    }
    return out;
}

static bool proposal_valid(const float * p) {
    for (int i = 0; i < 4; i++) {
        if (!(p[i] > 0.01f && p[i] < 0.99f)) {
            return false;
        }
    }
    return true;
}

std::vector<float> output_proposals_data_multilevel(const std::vector<std::pair<int, int>> & levels) {
    int64_t total = 0;
    for (auto & lv : levels) {
        total += (int64_t) lv.first * lv.second;
    }
    std::vector<float> out((size_t) total * 4);
    size_t off = 0;
    for (size_t lvl = 0; lvl < levels.size(); lvl++) {
        const int gw = levels[lvl].first, gh = levels[lvl].second;
        const float wh = 0.05f * (float) (1 << lvl); // 0.05 * 2^lvl
        for (int row = 0; row < gh; row++) {
            for (int col = 0; col < gw; col++) {
                float * p = &out[(off + (size_t) row * gw + col) * 4];
                p[0] = (col + 0.5f) / gw;
                p[1] = (row + 0.5f) / gh;
                p[2] = wh;
                p[3] = wh;
                if (!proposal_valid(p)) {
                    p[0] = p[1] = p[2] = p[3] = 0.0f;
                }
            }
        }
        off += (size_t) gw * gh;
    }
    return out;
}

std::vector<float> valid_mask_data_multilevel(const std::vector<std::pair<int, int>> & levels) {
    int64_t total = 0;
    for (auto & lv : levels) {
        total += (int64_t) lv.first * lv.second;
    }
    std::vector<float> mask((size_t) total);
    size_t off = 0;
    for (size_t lvl = 0; lvl < levels.size(); lvl++) {
        const int gw = levels[lvl].first, gh = levels[lvl].second;
        const float wh = 0.05f * (float) (1 << lvl);
        for (int row = 0; row < gh; row++) {
            for (int col = 0; col < gw; col++) {
                float p[4] = { (col + 0.5f) / gw, (row + 0.5f) / gh, wh, wh };
                mask[off + (size_t) row * gw + col] = proposal_valid(p) ? 1.0f : 0.0f;
            }
        }
        off += (size_t) gw * gh;
    }
    return mask;
}

// (cx,cy,w,h)-reparam box decode: cxcy = delta[:2]*base_wh + base_cxcy,
// wh = exp(delta[2:])*base_wh. delta/base are (4,T,N).
static ggml_tensor * bbox_reparam_decode(ggml_context * ctx, ggml_tensor * delta, ggml_tensor * base) {
    ggml_tensor * base_cxcy = ggml_cont(ctx, ggml_view_3d(ctx, base, 2, base->ne[1], base->ne[2], base->nb[1], base->nb[2], 0));
    ggml_tensor * base_wh   = ggml_cont(ctx, ggml_view_3d(ctx, base, 2, base->ne[1], base->ne[2], base->nb[1], base->nb[2], 2 * sizeof(float)));
    ggml_tensor * delta_cxcy = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, delta->ne[1], delta->ne[2], delta->nb[1], delta->nb[2], 0));
    ggml_tensor * delta_wh   = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, delta->ne[1], delta->ne[2], delta->nb[1], delta->nb[2], 2 * sizeof(float)));

    ggml_tensor * cxcy = ggml_add(ctx, ggml_mul(ctx, delta_cxcy, base_wh), base_cxcy);
    ggml_tensor * wh   = ggml_mul(ctx, ggml_exp(ctx, delta_wh), base_wh);
    return ggml_concat(ctx, cxcy, wh, 0);
}

// Same as bbox_reparam_decode, but assembled via ggml_set instead of
// ggml_concat -- CONCAT has NO backward case at all (same class of gap as
// ggml_norm/ggml_sigmoid; see docs/decisions/0003-training.md), which
// aborted the process the moment a real training step needed a gradient
// through pred_boxes = bbox_reparam_decode(bbox_embed_output, ref_points).
// SET's backward IS implemented. `base` is used as the destination
// template purely for its (4,T,N) shape/strides -- it never needs its own
// gradient here (ref_points isn't a trainable tensor in the detection-
// head-only scope), so ggml's dependency analysis skips computing one for
// it, same as layer_norm_affine_diff leaving layer_norm_affine untouched
// for inference-only paths. Numerically identical to bbox_reparam_decode;
// use that one for inference/frozen paths (marginally cheaper, one op
// instead of two), this one wherever the result must flow into backward.
static ggml_tensor * bbox_reparam_decode_diff(ggml_context * ctx, ggml_tensor * delta, ggml_tensor * base) {
    ggml_tensor * base_cxcy = ggml_cont(ctx, ggml_view_3d(ctx, base, 2, base->ne[1], base->ne[2], base->nb[1], base->nb[2], 0));
    ggml_tensor * base_wh   = ggml_cont(ctx, ggml_view_3d(ctx, base, 2, base->ne[1], base->ne[2], base->nb[1], base->nb[2], 2 * sizeof(float)));
    ggml_tensor * delta_cxcy = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, delta->ne[1], delta->ne[2], delta->nb[1], delta->nb[2], 0));
    ggml_tensor * delta_wh   = ggml_cont(ctx, ggml_view_3d(ctx, delta, 2, delta->ne[1], delta->ne[2], delta->nb[1], delta->nb[2], 2 * sizeof(float)));

    ggml_tensor * cxcy = ggml_add(ctx, ggml_mul(ctx, delta_cxcy, base_wh), base_cxcy);
    // clamp delta_wh before exp(): an untrained/frozen head fed
    // out-of-distribution input can produce a large enough delta_wh for
    // exp() to overflow to +inf, propagating into the loss as garbage --
    // +-4 covers any realistic trained-model box-size ratio (e^4 =~ 55x)
    // with margin.
    ggml_tensor * wh = ggml_mul(ctx, ggml_exp(ctx, clamp_diff(ctx, delta_wh, -4.0f, 4.0f)), base_wh);

    ggml_tensor * result = ggml_set(ctx, base, cxcy, base->nb[1], base->nb[2], base->nb[3], 0);
    return ggml_set(ctx, result, wh, base->nb[1], base->nb[2], base->nb[3], 2 * sizeof(float));
}

// gen_sineembed_for_position, one coordinate: v is (1,n_query,1); sine_scale
// is the GGUF constant (1,dim) = 2*pi/dim_t. Returns (dim,n_query,1) with
// interleaved sin(raw[2k])/cos(raw[2k+1]).
static ggml_tensor * sine_embed_1coord(ggml_context * ctx, ggml_tensor * v, ggml_tensor * sine_scale, int dim) {
    const int64_t n_query = v->ne[1];
    ggml_tensor * v2 = ggml_reshape_2d(ctx, ggml_cont(ctx, v), 1, n_query);              // (1,n_query)
    ggml_tensor * raw = ggml_mul_mat(ctx, sine_scale, v2);                               // (dim,n_query)

    ggml_tensor * pairs = ggml_reshape_3d(ctx, raw, 2, dim / 2, n_query);                // (2,dim/2,n_query)
    ggml_tensor * even = ggml_cont(ctx, ggml_view_3d(ctx, pairs, 1, dim / 2, n_query, pairs->nb[1], pairs->nb[2], 0));
    ggml_tensor * odd  = ggml_cont(ctx, ggml_view_3d(ctx, pairs, 1, dim / 2, n_query, pairs->nb[1], pairs->nb[2], sizeof(float)));
    ggml_tensor * s = ggml_sin(ctx, even);
    ggml_tensor * c = ggml_cos(ctx, odd);
    ggml_tensor * interleaved = ggml_concat(ctx, s, c, 0); // (2,dim/2,n_query)
    return ggml_reshape_3d(ctx, interleaved, dim, n_query, 1);
}

DecoderOutput rfdetr_decoder(Model & m, ggml_tensor * memory, const DecoderParams & p,
                             ggml_tensor * topk_idx_override,
                             const KeypointParams * kp, ggml_tensor * kp_memory, bool trainable) {
    ggml_context * ctx = m.ctx_g;
    // Every LayerNorm in this function uses the diff (primitive-decomposed,
    // backward-capable) version when trainable is set, matching the
    // same flag that already selects bbox_reparam_decode_diff -- gradient
    // must flow through EVERY norm between the trainable weights (whichever
    // subset the caller marks with ggml_set_param) and pred_boxes/
    // pred_logits, regardless of which specific decoder weights end up
    // trainable. layer_norm_affine_diff is numerically identical to
    // layer_norm_affine for inference (tests/test_norm_backward.cpp), so
    // this has zero effect on non-trainable graphs.
    auto norm = [&](ggml_tensor * x, const std::string & pre) {
        return trainable ? layer_norm_affine_diff(m, x, pre) : layer_norm_affine(m, x, pre);
    };
    const bool multilevel = !p.levels.empty();
    int gwh = p.gw * p.gh;
    if (multilevel) {
        gwh = 0;
        for (auto & lv : p.levels) {
            gwh += lv.first * lv.second;
        }
    }
    const int hd = p.hidden_dim;

    ggml_tensor * proposals = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, gwh, 1);
    ggml_set_name(proposals, "output_proposals");
    ggml_set_input(proposals);
    // Also protect from the graph allocator's buffer-reuse: `ggml_set_input`
    // alone does NOT stop a leaf tensor's buffer from being handed to a
    // later node's output -- only `ggml_set_output` does that (see
    // ggml-alloc.c's `ggml_gallocr_free_node`, which only special-cases
    // GGML_TENSOR_FLAG_OUTPUT). Worse, leaf tensors like this one aren't
    // reachable via `ggml_graph_node()`/`ggml_graph_n_nodes()` (those only
    // iterate computed nodes), so a "protect every node" sweep silently
    // misses them -- this is exactly what let the RFDETRSegXLarge
    // divergence (docs/decisions/0001-open-work.md) survive an earlier
    // blanket-protection experiment: found by tracing `topk_override`
    // (the analogous caller-created leaf in tests/test_segmentation_xlarge.cpp)
    // reading back as pure garbage int32 values on that graph's 18819+ nodes,
    // then finding this tensor has the exact same unprotected-leaf shape.
    ggml_set_output(proposals);

    // Multi-level only: mask `memory` (zero tokens whose grid-cell proposal
    // fell outside (0.01,0.99), see output_proposals_data_multilevel) before
    // the two-stage proposal heads -- matches upstream's
    // output_memory.masked_fill(~valid,0). Deformable cross-attention below
    // still reads the RAW (unmasked) `memory`, matching upstream exactly.
    ggml_tensor * valid_mask = nullptr;
    ggml_tensor * memory_for_proposals = memory;
    if (multilevel) {
        valid_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, gwh, 1);
        ggml_set_name(valid_mask, "valid_mask");
        ggml_set_input(valid_mask);
        ggml_set_output(valid_mask); // same leaf-buffer-reuse protection as `proposals` above
        memory_for_proposals = ggml_mul(ctx, memory, valid_mask); // broadcasts (1,gwh,1) over (hd,gwh,1)
    }

    // two-stage encoder proposals (group 0 only; group_detr is a no-op at inference)
    ggml_tensor * output_memory = norm(linear(m, memory_for_proposals, "enc_output.0"), "enc_output_norm.0");
    ggml_tensor * enc_class = linear(m, output_memory, "enc_out_class_embed.0");     // (num_classes, gwh, 1)
    ggml_tensor * enc_delta = mlp(m, output_memory, "enc_out_bbox_embed.0", 3);      // (4, gwh, 1)
    ggml_tensor * enc_boxes = bbox_reparam_decode(ctx, enc_delta, proposals);        // (4, gwh, 1)

    // top-k by per-location max class score (no reduce-max-along-ne0 op --
    // max-pool with a kernel spanning the whole class axis instead)
    ggml_tensor * topk_idx;
    if (topk_idx_override) {
        topk_idx = topk_idx_override;
    } else {
        ggml_tensor * score = ggml_pool_1d(ctx, enc_class, GGML_OP_POOL_MAX, p.num_classes, p.num_classes, 0); // (1,gwh,1)
        ggml_tensor * score1d = ggml_reshape_1d(ctx, ggml_cont(ctx, score), gwh);
        // argsort_top_k, not top_k: top_k's indices are documented as unordered and the CPU
        // kernel swaps the first two, so query slot order would differ by backend.
        topk_idx = ggml_reshape_1d(ctx, ggml_argsort_top_k(ctx, score1d, p.num_queries), p.num_queries); // I32 (num_queries)
    }

    ggml_tensor * ts_boxes = ggml_get_rows(ctx, ggml_reshape_2d(ctx, enc_boxes, 4, gwh), topk_idx); // (4,num_queries)
    ts_boxes = ggml_reshape_3d(ctx, ts_boxes, 4, p.num_queries, 1);

    // learned static content queries + per-query reference-point delta
    ggml_tensor * query_feat_w = m.get("query_feat.weight");     // (hidden_dim, num_queries*group_detr)
    ggml_tensor * refpt_w      = m.get("refpoint_embed.weight"); // (4, num_queries*group_detr)
    ggml_tensor * tgt = ggml_cont(ctx, ggml_view_2d(ctx, query_feat_w, hd, p.num_queries, query_feat_w->nb[1], 0));
    tgt = ggml_reshape_3d(ctx, tgt, hd, p.num_queries, 1);
    ggml_tensor * subset = ggml_cont(ctx, ggml_view_2d(ctx, refpt_w, 4, p.num_queries, refpt_w->nb[1], 0));
    subset = ggml_reshape_3d(ctx, subset, 4, p.num_queries, 1);

    ggml_tensor * ref_points = bbox_reparam_decode(ctx, subset, ts_boxes); // (4, num_queries, 1) -- fixed for every layer

    // query positional embedding (once, lite_refpoint_refine). Sine dim is
    // d_model/2 per coordinate (gen_sineembed_for_position is called with
    // self.d_model // 2 upstream, NOT its default dim=128 -- 128 only
    // happens to equal 256/2, which every variant validated before
    // RFDETRLargeDeprecated's hidden_dim=384 shared) so 4 coords concat to
    // 4*(hd/2) = 2*hd, matching ref_point_head's MLP(2*d_model, ...) input.
    const int sine_dim = hd / 2;
    ggml_tensor * sine_scale = m.get("decoder.sine_scale"); // (1,sine_dim) constant, GGUF-baked: 2*pi/dim_t
    auto coord = [&](int off) {
        return ggml_cont(ctx, ggml_view_3d(ctx, ref_points, 1, p.num_queries, 1, ref_points->nb[1], ref_points->nb[2], (size_t) off * sizeof(float)));
    };
    ggml_tensor * pos_y = sine_embed_1coord(ctx, coord(1), sine_scale, sine_dim);
    ggml_tensor * pos_x = sine_embed_1coord(ctx, coord(0), sine_scale, sine_dim);
    ggml_tensor * pos_w = sine_embed_1coord(ctx, coord(2), sine_scale, sine_dim);
    ggml_tensor * pos_h = sine_embed_1coord(ctx, coord(3), sine_scale, sine_dim);
    ggml_tensor * sine = ggml_concat(ctx, ggml_concat(ctx, pos_y, pos_x, 0), ggml_concat(ctx, pos_w, pos_h, 0), 0); // (2*hd,nq,1)
    ggml_tensor * query_pos = mlp(m, sine, "decoder.ref_point_head", 2); // (hd,nq,1), ReLU between layers 0/1

    // keypoint init (once, decoder-level ConditionalQueryInitializer conditioned
    // on the initial tgt -- before any decoder layer runs)
    ggml_tensor * keypoint_tgt = nullptr;
    ggml_tensor * keypoint_pos = nullptr;
    if (kp) {
        ggml_tensor * cond = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, tgt, 0, 2, 1, 3)), hd, 1, p.num_queries);
        keypoint_tgt = conditional_query_initializer(m, cond, "keypoint_query_initializer", kp->num_kp, hd);
        keypoint_pos = ggml_reshape_3d(ctx, m.get("decoder.keypoint_pos_embed"), hd, kp->num_kp, 1);
    }

    // decoder layers
    ggml_tensor * tgt_cur = tgt;
    std::vector<ggml_tensor *> out_hs;
    std::vector<ggml_tensor *> out_kp;
    for (int l = 0; l < p.dec_layers; l++) {
        const std::string pre = "decoder.layers." + std::to_string(l) + ".";

        // Self-attention with v computed from tgt_cur (not tgt_cur+query_pos) --
        // ops.h's generic self_attn() applies q/k/v to the same input, which
        // doesn't fit nn.MultiheadAttention(query=tgt+pos, key=tgt+pos, value=tgt)
        // here, so it's inlined instead.
        ggml_tensor * q_in = ggml_add(ctx, tgt_cur, query_pos);
        const int64_t T = tgt_cur->ne[1];
        ggml_tensor * qv = linear(m, q_in, pre + "self_attn.q_proj");
        ggml_tensor * kv = linear(m, q_in, pre + "self_attn.k_proj");
        ggml_tensor * vv = linear(m, tgt_cur, pre + "self_attn.v_proj");
        const int64_t hdh = hd / p.sa_nheads;
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, qv, hdh, p.sa_nheads, T, 1), 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, kv, hdh, p.sa_nheads, T, 1), 0, 2, 1, 3));
        ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, vv, hdh, p.sa_nheads, T, 1), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, kh, qh), 1.0f / sqrtf((float) hdh)));
        ggml_tensor * kqv = ggml_mul_mat(ctx, vh, kq);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_3d(ctx, kqv, hd, T, 1);
        ggml_tensor * self_out = linear(m, kqv, pre + "self_attn.out_proj");

        tgt_cur = norm(ggml_add(ctx, tgt_cur, self_out), pre + "norm1");

        ggml_tensor * cross_q = ggml_add(ctx, tgt_cur, query_pos);
        ggml_tensor * cross_out = multilevel
            ? ms_deform_attn_multilevel(m, cross_q, memory, ref_points, pre + "cross_attn",
                                        p.ca_nheads, p.dec_n_points, p.levels)
            : ms_deform_attn(m, cross_q, memory, ref_points, pre + "cross_attn",
                             p.ca_nheads, p.dec_n_points, p.gw, p.gh);
        tgt_cur = norm(ggml_add(ctx, tgt_cur, cross_out), pre + "norm2");

        ggml_tensor * ff = linear(m, tgt_cur, pre + "linear1");
        ff = ggml_relu(ctx, ff);
        ff = linear(m, ff, pre + "linear2");
        tgt_cur = norm(ggml_add(ctx, tgt_cur, ff), pre + "norm3");

        // GroupPose keypoint sublayer (4th sublayer, only when kp is set): runs
        // after the ordinary 3 sublayers, using the SAME per-layer query_pos and
        // fixed ref_points; MUTATES tgt_cur further (kp_inst_norm's output, not
        // norm3's) -- both the next layer and this layer's decoder.norm tap below
        // must see the keypoint-updated tgt_cur, matching upstream's forward_post
        // returning (tgt, keypoint_tgt) as one combined step.
        if (kp) {
            KeypointLayerOutput kout = keypoint_decoder_layer(m, tgt_cur, query_pos, keypoint_tgt, keypoint_pos,
                                                              ref_points, kp_memory, pre, *kp, p.gw, p.gh);
            tgt_cur = kout.tgt;
            keypoint_tgt = kout.keypoint_tgt;
            out_kp.push_back(keypoint_tgt);
        }

        // return_intermediate=True always: the shared decoder.norm is applied to
        // EVERY layer's output independently for the intermediate/aux list (the
        // raw, unnormed tgt_cur is what continues into the next layer, unaffected).
        // Only the last layer's normed output feeds pred_boxes/pred_logits, but the
        // segmentation head needs all of them (one DepthwiseConvBlock per layer).
        out_hs.push_back(norm(tgt_cur, "decoder.norm"));
    }

    ggml_tensor * final_hs = out_hs.back();

    DecoderOutput out;
    ggml_tensor * class_logits = linear(m, final_hs, "class_embed");
    ggml_tensor * bbox_delta = mlp(m, final_hs, "bbox_embed", 3);
    // trainable selects backward-capable (clamped, diverges from
    // upstream only when delta_wh is extreme) vs the exact-match default
    // -- see decoder.h's docstring and docs/decisions/0001-open-work.md's
    // SegXLarge root-cause writeup.
    out.pred_boxes = trainable ? bbox_reparam_decode_diff(ctx, bbox_delta, ref_points)
                                     : bbox_reparam_decode(ctx, bbox_delta, ref_points);
    out.output_proposals = proposals;
    out.valid_mask = valid_mask;
    out.hidden_states = out_hs;
    out.dbg_ts_boxes = ts_boxes;
    out.dbg_ref_points = ref_points;
    out.dbg_enc_boxes = enc_boxes;
    out.dbg_enc_delta = enc_delta;
    out.dbg_topk_idx = topk_idx;
    out.dbg_bbox_delta = bbox_delta;
    out.dbg_final_hs = final_hs;

    if (kp) {
        KeypointHeadOutput kdec = keypoint_final_decode(m, out_kp.back(), ref_points, kp->num_kp,
                                                        kp->num_keypoint_classes, kp->active_class_idx);
        out.pred_keypoints = kdec.pred_keypoints;
        // class-logit boost applies only to kp->active_class_idx (this
        // checkpoint's single active keypoint class, confirmed against the
        // checkpoint's own _kp_active_mask buffer -- see
        // docs/decisions/keypoints.md); every other class (including
        // background) is unchanged.
        const size_t boost_off = (size_t) kp->active_class_idx * sizeof(float);
        ggml_tensor * class_col = ggml_cont(ctx, ggml_view_3d(ctx, class_logits, 1, class_logits->ne[1], 1,
                                                              class_logits->nb[1], class_logits->nb[2], boost_off));
        class_col = ggml_add(ctx, class_col, kdec.class_logit_boost); // both (1, num_queries, 1)
        class_logits = ggml_set_2d(ctx, class_logits, ggml_reshape_2d(ctx, class_col, 1, p.num_queries),
                                   class_logits->nb[1], boost_off);
    }
    out.pred_logits = class_logits;
    return out;
}

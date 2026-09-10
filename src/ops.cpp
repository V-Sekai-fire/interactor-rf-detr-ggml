#include "ops.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

Model::~Model() {
    for (ggml_backend_buffer * b : bufs) ggml_backend_buffer_free(b);
    for (ggml_context * c : ctx_w) ggml_free(c);
}

static void collect_config_kv(gguf_context * g, std::map<std::string, std::string> & config_json) {
    for (int64_t i = 0; i < gguf_get_n_kv(g); i++) {
        const char * key = gguf_get_key(g, i);
        if (gguf_get_kv_type(g, i) != GGUF_TYPE_STRING) continue;
        std::string k = key;
        for (const char * suf : { ".config_json" }) {
            size_t n = strlen(suf);
            if (k.size() > n && k.compare(k.size() - n, n, suf) == 0) {
                config_json[k] = gguf_get_val_str(g, i);
            }
        }
    }
}

bool Model::load(const char * path) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &c };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    collect_config_kv(g, config_json);
    for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        weights[ggml_get_name(t)] = t;
    }
    ctx_w.push_back(c);
    gguf_free(g);
    return true;
}

bool Model::load_backend(const char * path, ggml_backend_buffer_type * buft) {
    ggml_context * c = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &c };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    collect_config_kv(g, config_json);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!buf) { gguf_free(g); return false; }
    bufs.push_back(buf);

    FILE * f = fopen(path, "rb");
    if (!f) { gguf_free(g); return false; }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<uint8_t> staging;
    for (ggml_tensor * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
        int64_t idx = gguf_find_tensor(g, ggml_get_name(t));
        if (idx < 0) { fclose(f); gguf_free(g); return false; }
        size_t nbytes = ggml_nbytes(t);
        staging.resize(nbytes);
#ifdef _WIN32
        int seek_rc = _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET);
#else
        int seek_rc = fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(g, idx)), SEEK_SET);
#endif
        if (seek_rc != 0 || fread(staging.data(), 1, nbytes, f) != nbytes) {
            fclose(f); gguf_free(g); return false;
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        weights[ggml_get_name(t)] = t;
    }
    fclose(f);
    ctx_w.push_back(c);
    gguf_free(g);
    return true;
}

ggml_tensor * Model::get(const std::string & name) const {
    auto it = weights.find(name);
    // A caller embedding this in a host process (a Godot module) must be able to
    // report a bad checkpoint, not be terminated by it.
    if (it == weights.end()) {
        missing.push_back(name);
        return nullptr;
    }
    return it->second;
}

bool Model::has(const std::string & name) const {
    return weights.count(name) != 0;
}

ggml_tensor * conv2d(Model & m, ggml_tensor * x, const std::string & pre, int stride, int pad) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * w = m.get(pre + ".weight");
    ggml_type it = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im = ggml_im2col(ctx, w, x, stride, stride, pad, pad, 1, 1, true, it);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im, im->ne[0], im->ne[3] * im->ne[2] * im->ne[1]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]));
    r = ggml_reshape_4d(ctx, r, im->ne[1], im->ne[2], im->ne[3], w->ne[3]);
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 1, 3, 2));
    if (m.has(pre + ".bias")) {
        ggml_tensor * b = ggml_reshape_4d(ctx, m.get(pre + ".bias"), 1, 1, w->ne[3], 1);
        r = ggml_add(ctx, r, b);
    }
    return r;
}

ggml_tensor * conv_transpose2d(Model & m, ggml_tensor * x, const std::string & pre, int stride) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * w = m.get(pre + ".weight");
    ggml_tensor * r = ggml_conv_transpose_2d_p0(ctx, w, x, stride);
    if (m.has(pre + ".bias")) {
        ggml_tensor * b = ggml_reshape_4d(ctx, m.get(pre + ".bias"), 1, 1, w->ne[2], 1);
        r = ggml_add(ctx, r, b);
    }
    return r;
}

ggml_tensor * layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre, float eps) {
    ggml_context * ctx = m.ctx_g;
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, m.get(pre + ".weight"));
    return ggml_add(ctx, x, m.get(pre + ".bias"));
}

ggml_tensor * layer_norm_affine_diff(Model & m, ggml_tensor * x, const std::string & pre, float eps) {
    // ggml_norm has no backward case (see docs/decisions/0003-training.md), so
    // this rebuilds LayerNorm from primitives that do. Two of ggml's binary
    // backward rules are asymmetric: SUB and DIV only propagate a gradient to
    // src1 when src1 is the SAME shape as the output (no repeat_back), while
    // ADD and MUL handle a smaller/broadcast src1 correctly. So every
    // broadcasting subtract is written as add(x, scale(mean,-1)) and every
    // broadcasting divide as mul(x, reciprocal) (reciprocal computed via a
    // same-shape, non-broadcast div first). ggml_mean's own backward also
    // only works when its output is a true scalar (it reuses ADD1, which
    // asserts a scalar operand) -- useless here since mean's output is
    // (1,T,N), not (1,1,1,1). ggml_sum_rows has the same forward shape as
    // ggml_mean over ne[0] but a shape-general backward (plain repeat), so
    // mean is rebuilt as scale(sum_rows(x), 1/dim) instead.
    ggml_context * ctx = m.ctx_g;
    const float inv_dim = 1.0f / (float) x->ne[0];
    ggml_tensor * mean = ggml_scale(ctx, ggml_sum_rows(ctx, x), inv_dim); // (1, T, N)
    ggml_tensor * centered = ggml_add(ctx, x, ggml_scale(ctx, mean, -1.0f)); // broadcast add, has backward
    ggml_tensor * var = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, centered)), inv_dim); // (1, T, N)
    ggml_tensor * std = ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, eps));
    ggml_tensor * ones = ggml_scale_bias(ctx, var, 0.0f, 1.0f);          // (1, T, N), all ones
    ggml_tensor * inv_std = ggml_div(ctx, ones, std);                    // same shape, non-broadcast div
    ggml_tensor * normed = ggml_mul(ctx, centered, inv_std);             // broadcast mul, has backward
    normed = ggml_mul(ctx, normed, m.get(pre + ".weight"));
    return ggml_add(ctx, normed, m.get(pre + ".bias"));
}

ggml_tensor * linear(Model & m, ggml_tensor * x, const std::string & pre) {
    ggml_context * ctx = m.ctx_g;
    ggml_tensor * y = ggml_mul_mat(ctx, m.get(pre + ".weight"), x);
    if (m.has(pre + ".bias")) y = ggml_add(ctx, y, m.get(pre + ".bias"));
    return y;
}

ggml_tensor * clamp_diff(ggml_context * ctx, ggml_tensor * x, float lo, float hi) {
    ggml_tensor * a = ggml_scale_bias(ctx, ggml_relu(ctx, ggml_scale_bias(ctx, x, 1.0f, -lo)), 1.0f, lo);
    return ggml_sub(ctx, a, ggml_relu(ctx, ggml_scale_bias(ctx, x, 1.0f, -hi)));
}

ggml_tensor * mlp(Model & m, ggml_tensor * x, const std::string & pre, int n_layers) {
    ggml_context * ctx = m.ctx_g;
    for (int i = 0; i < n_layers; i++) {
        x = linear(m, x, pre + ".layers." + std::to_string(i));
        if (i + 1 < n_layers) x = ggml_relu(ctx, x);
    }
    return x;
}

ggml_tensor * self_attn(Model & m, ggml_tensor * x, const std::string & pre, int n_head) {
    ggml_context * ctx = m.ctx_g;
    const int64_t C = x->ne[0], T = x->ne[1], N = x->ne[2];
    const int64_t hd = C / n_head;

    ggml_tensor * q = linear(m, x, pre + ".q_proj");
    ggml_tensor * k = linear(m, x, pre + ".k_proj");
    ggml_tensor * v = linear(m, x, pre + ".v_proj");

    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_head, T, N), 0, 2, 1, 3)); // (hd,T,H,N)
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_head, T, N), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_head, T, N), 1, 2, 0, 3)); // (T,hd,H,N)

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                       // (T,T,H,N)
    kq = ggml_soft_max(ctx, ggml_scale(ctx, kq, 1.0f / sqrtf((float) hd)));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                     // (hd,T,H,N)
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));         // (hd,H,T,N)
    kqv = ggml_reshape_3d(ctx, kqv, C, T, N);

    return linear(m, kqv, pre + ".out_proj");
}

ggml_tensor * spatial_layer_norm_affine(Model & m, ggml_tensor * x, const std::string & pre, float eps) {
    ggml_context * ctx = m.ctx_g;
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], N = x->ne[3];
    ggml_tensor * cf = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)); // (C,W,H,N), C fastest
    cf = ggml_reshape_3d(ctx, cf, C, W * H, N);
    cf = ggml_norm(ctx, cf, eps);
    cf = ggml_mul(ctx, cf, m.get(pre + ".weight"));
    cf = ggml_add(ctx, cf, m.get(pre + ".bias"));
    cf = ggml_reshape_4d(ctx, cf, C, W, H, N);
    return ggml_cont(ctx, ggml_permute(ctx, cf, 2, 0, 1, 3)); // back to (W,H,C,N)
}

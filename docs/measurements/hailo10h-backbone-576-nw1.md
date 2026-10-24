# Compiled layer census: backbone, 576, `num_windows=1`, Hailo-10H

**The decision this supports is not here.** RFD 107e decides how the chain deploys to an
edge accelerator, and `weftspun/logbook`'s `logbook-edge-npu-and-the-anny-forward.md`
carries the measurements and the two retractions behind it. RFD66606-1000 asks a document to
point at its source rather than copy it, and `docs/decisions/0005-hailo-edge-compile.md`
copied both. It was removed rather than maintained in three places.

What stays here is what belongs beside the code: the census of the graph this repository's
own scripts produce, and the compiled graph itself.

Clipped from the HAR that DFC 5.3.0 produced. `hailo10h-backbone-576-nw1.hn.zst`
beside this file is the compiled graph itself, 6 KB.

The full HAR is 102 MB and is not kept, because 101.4 MB of it is
`windows_1.npz` -- the weights, which are a copy of a checkpoint already held. zstd
does not rescue that: float32 weights are high entropy and compress 8.5%, 102.5 MB to
93.8 MB, with `-19` no better than `-3`. The `.hn` is the part the compiler actually
produced and it compresses 41.8x, 246 KB to 6 KB.

Reproduce with `deploy/hailo-dfc/build.sh 576` then `gate_dfc_parse.py`. Read the
committed graph with `zstd -dc hailo10h-backbone-576-nw1.hn.zst | python -m json.tool`.

    ONNX in   825 nodes, 23 distinct operators
    HN out    336 layers
    arch      hailo10h, DFC 5.3.0

| count | Hailo layer type    |
| ----- | ------------------- |
| 130   | normalization       |
| 65    | conv                |
| 46    | layer_normalization |
| 25    | ew_add              |
| 24    | matmul              |
| 14    | feature_splitter    |
| 12    | softmax             |
| 5     | format_conversion   |
| 4     | concat              |
| 4     | slice               |
| 2     | const_input         |
| 2     | shortcut            |
| 2     | output_layer        |
| 1     | input_layer         |

Two things worth reading off this rather than inferring.

**`layer_normalization` survives as a first-class layer, 46 of them.** The
`IndexError` in LayerNorm axis mapping that the unfolded graph raises is therefore not
LayerNorm being unsupported. It is a downstream symptom, which is what
`0005-hailo-edge-compile.md` records.

**4 concat layers compile.** Concat is not unsupported either. The rejected one was
concat over the batch axis specifically, which is what windowing produces.

Not measured here: cycles, latency, or utilisation. Those need the DFC profiler, and
every throughput figure quoted elsewhere assumes a 30% utilisation that nothing has
checked.

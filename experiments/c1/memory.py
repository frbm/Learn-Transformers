"""C1 Section 1: predicted against measured peak memory for five configurations."""
import argparse

from lm.train.memory import activation_bytes, logits_bytes, measure_peak_memory, predict_peak_memory, static_bytes


def main():
    from lm.model import Config

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="", help="dump a snapshot of the first configuration to this pickle")
    args = ap.parse_args()
    G = 2**30
    rows = [  # (shape, B, T, checkpointing, ce_chunk)
        (dict(d_model=256, n_layers=4, n_heads=4, n_kv_heads=4), 32, 512, False, None),
        (dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4), 16, 1024, False, None),
        (dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4), 16, 1024, False, 2048),
        (dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4), 16, 1024, True, None),
        (dict(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4), 16, 1024, True, 2048),
    ]
    print(f"{'config':34s} {'static':>7s} {'act':>7s} {'logits':>7s} {'pred':>7s} {'meas':>7s} {'err':>6s}")
    for i, (shape, B, T, ck, ch) in enumerate(rows):
        cfg = Config(vocab_size=16448, max_seq_len=T, checkpointing=ck, **shape)
        pred = predict_peak_memory(cfg, B, T, checkpointing=ck, ce_chunk=ch)
        meas = measure_peak_memory(cfg, B, T, ck, ch, snapshot=(args.snapshot if i == 0 else None) or None)
        name = f"d{cfg.d_model} L{cfg.n_layers} {B}x{T} ckpt={int(ck)} chunk={ch}"
        print(f"{name:34s} {static_bytes(cfg) / G:7.3f} {activation_bytes(cfg, B, T, ck) / G:7.3f} "
              f"{logits_bytes(cfg, B, T, ch) / G:7.3f} {pred / G:7.3f} {meas / G:7.3f} {(pred - meas) / meas:+6.1%}")


if __name__ == "__main__":
    main()

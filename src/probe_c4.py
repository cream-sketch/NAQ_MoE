from datasets import load_dataset


def main() -> None:
    candidates = [
        ("allenai/c4", "en", "validation"),
        ("allenai/c4", "en", "train"),
        ("c4", "en", "validation"),
    ]
    for name, config, split in candidates:
        try:
            ds = load_dataset(name, config, split=split, streaming=True)
            ex = next(iter(ds))
            print("OK", name, config, split, list(ex.keys()), repr((ex.get("text") or "")[:120]))
            return
        except Exception as exc:
            print("ERR", name, config, split, type(exc).__name__, str(exc)[:300])
    raise SystemExit(1)


if __name__ == "__main__":
    main()

[Documentation home](README.md)

# Files, formats, and run identity

This page owns the persistence story: the plain-data form produced by `to_dict`, JSON and optional YAML, extension dispatch, load-time document rules, atomic saves, and stable run identifiers.


## The persistence pipeline

Loading runs `file -> parser -> mapping -> from_dict -> config object`. Saving runs the reverse: `config object -> to_dict -> formatter -> atomic write`.

Both formats share the same plain-data model (mappings, sequences, strings, numbers, booleans, `None`), so everything on this page applies to JSON and YAML alike.


## Plain data with `to_dict`

- Dataclasses become dicts in field-declaration order.
- Enums become their `.value`.
- `Path`, `datetime`, `date`, and `time` become strings (`str(path)`, `.isoformat()`).
- `list`, `tuple`, `set`, and `frozenset` all become lists; sets are ordered by each element's canonical JSON text, so output is deterministic across processes and `PYTHONHASHSEED` values.
- Values outside the plain-data model (for example `Decimal`, or a non-finite float) raise `ConfigError`.


## JSON

```python
config.save_json("resolved.json")        # atomic write, returns Path
text = config.dumps_json()               # indent=2, trailing newline
config = TrainingConfig.load_json(path)  # ConfigError on any failure
```

Output uses two-space indentation by default (`indent` parameter), escapes text to ASCII, keeps field-declaration order, and ends with a trailing newline.


## YAML

YAML support ships with the base install, alongside the JSON loaders:

```python
config = TrainingConfig.load_yaml("experiment.yaml")
config.save_yaml("resolved.yaml")
```

- Loading uses `safe_load` and saving uses `safe_dump`, restricted to the shared plain-data model.
- `sort_keys=False` is the default, so YAML output preserves field-declaration order; pass `sort_keys=True` for alphabetical keys.


## Extension dispatch

`from_file` / `to_file` (and the matching `ConfigRoot` methods) route by file suffix: `.json`, `.yaml`, and `.yml`, case-insensitive.

```python
config = TrainingConfig.from_file("experiment.yaml")
config.to_file(run_dir / "resolved.json")
```

A path with a missing or unrecognized suffix raises `ConfigError` naming the supported extensions.


## Document and read rules

- Files are read as UTF-8; read failures and decode failures raise `ConfigError` with the file's context.
- Malformed syntax (JSON or YAML) raises `ConfigError`.
- A top-level mapping feeds `from_dict`.
- A parsed null document (JSON `null`, YAML `null`, or an empty YAML file) feeds `{}`, so a schema whose fields all carry defaults loads as pure defaults and required values are reported at their nested dotted paths. An empty JSON file is malformed JSON and raises accordingly.
- Any other top-level shape (a list, a scalar) raises `ConfigError` asking for a mapping document.


## Atomic writes

Every save funnels through one atomic writer:

- Parent directories are created as needed.
- Content goes to a uniquely named temporary sibling (exclusive creation), then renames over the destination, so readers see either the old file or the complete new one.
- On overwrite the destination's existing file mode is preserved; new files get the current umask's default. Unrelated `.tmp` files in the directory are left in place, and the writer's own temporary is removed if the write fails.
- Save functions return the destination `Path`.


## Resolved snapshots

Saving serializes the object as currently held in memory: defaults filled in, plus any programmatic changes made after loading. Writing `runs/<run_id>/resolved.json` therefore records exactly what the run used, whichever partial file or code path produced it.


## Stable run identity

`config_hash` fingerprints the resolved config: the leading `length` hex characters (12 by default, up to the full 64) of SHA-256 over the canonical compact JSON of the config's hashing fields. Equal resolved configs produce equal hashes across processes and `PYTHONHASHSEED` values, because the canonical JSON depends only on the resolved values, which makes the hash a natural run-directory name for sweeps:

```python
run_id = config.config_hash()          # "8e6ea26c7116"

for overrides in sweep:
    config = ExperimentConfig.from_dict({**base, **overrides})
    run_dir = Path("runs") / config.config_hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(run_dir / "resolved.json")
```

The digest rules, the `compare` and `hash` field projections, and array and tensor hashing live in [Equality and hashing](equality-and-hashing.md#stable-run-identity).


## Cross-format round trip

One config saved to JSON and to YAML loads back into equal dataclass trees: both formats carry the same plain data, temporal fields travel as ISO 8601 strings, and container annotations rebuild their exact types (`tuple[int, ...]` comes back as a tuple).

Container identity is restored by the annotation on re-load. In the raw file, tuples and sets appear as arrays, and `Any`-held tuple or set values stay lists after a round trip. See [`Any` and plain data](types-and-coercion.md#any-and-plain-data).


---

[Previous: Validation and errors](validation-and-errors.md) | [Home](README.md) | [Next: API reference](api-reference.md)

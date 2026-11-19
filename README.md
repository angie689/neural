# neural

Experimental **standalone Python → Apple Core ML** bridge for iPhone/iPad and macOS.

This repo does **not** modify or rebuild PyPhone. It uses the pure-Python [Rubicon-ObjC](https://github.com/beeware/rubicon-objc) package and `ctypes` to access the system Core ML framework at runtime. **No `_neural.m` or `PyImport_AppendInittab` is needed.**

## Install

```sh
pip install rubicon-objc
pip install git+https://github.com/angie689/neural.git
```

In PyPhone, install `rubicon-objc` first if Git dependency installation is not supported; copy the `neural` folder into a Python importable directory.

## Basic example

```python
import neural
print(neural.available())  # True means Core ML classes are accessible

model = neural.load("TestLinear.mlmodelc")
print(model.inputs, model.outputs)
print(model.predict({"input": [1.0]}))
```

Models can use `.mlmodelc`, `.mlmodel`, or `.mlpackage`. The last two are compiled on device using Core ML; this may fail in a restricted iOS host.

Optional `compute_units`: `"all"` (default; includes Neural Engine), `"cpu"`, `"cpu_gpu"`, or `"cpu_ne"` (iOS 16+).

## Current limits

- **UNTESTED ON DEVICE:** The Objective-C calls have not been executed on PyPhone or a physical iPhone. The code may need compatibility fixes. It is not yet verified that iOS allows access from the installed PyPhone build.
- **No automatic Apple Neural Engine guarantee:** Core ML chooses placement. `neural.available()` checks framework accessibility only, not ANE usage. `neural.neural_engine_available()` returns `None` rather than claiming detection.
- Multiarray, scalar, string inputs and numeric/array/dictionary outputs are attempted. Image and sequence inputs/outputs are not supported.
- Float16 multiarray inputs are not currently supported. Basic multiarray shapes assume fixed-size input descriptions.
- Objective-C methods use nil for NSError output; some failures may only have a generic error. Improper runtime or ABI support in the host may cause native exceptions.
- `rubicon-objc` requires an Objective-C runtime. This will **not** work on a normal Linux or Windows Python interpreter.
- The previous C/Objective-C extension implementation is **not** used in this standalone edition.

## Test progressively in PyPhone

```python
import neural
print("Core ML accessible:", neural.available())
```

Only if that returns `True`, test loading a model:

```python
model = neural.load("TestLinear.mlmodelc", compute_units="all")
print(model.inputs)
print(model.predict({"input": [1.0]}))
```

If `import rubicon.objc` fails, try `pip install rubicon-objc`. If Core ML cannot be loaded, this iOS Python host may forbid access, and a native app integration would then be necessary.

## Development

`python -m compileall neural` checks Python syntax. It does **not** validate native Core ML calls. The absence of a Mac/iPhone means there is no hardware verification yet.

### Why not just use the Objective-C C-API module?

A compiled `_neural.m` module must be linked and registered with a host's embedded interpreter. This package instead attempts the Objective-C calls dynamically so it can live in a separate GitHub repo and be imported with no app source edits.

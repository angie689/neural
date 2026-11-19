"""Standalone Core ML bridge for iOS, using Rubicon-ObjC (no PyPhone rebuild).

Experimental: requires a Python host that allows Objective-C runtime calls.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path


class NeuralError(RuntimeError):
    """Core ML could not load a model or perform a prediction."""


_COMPUTE = {"all": 2, "cpu": 0, "cpu_gpu": 1, "cpu_ne": 3}
# MLFeatureType values from CoreML/MLFeatureValue.h
_INT, _DOUBLE, _STRING, _IMAGE, _ARRAY, _DICT, _SEQ = 1, 2, 3, 4, 5, 6, 7
# MLMultiArrayDataType values from CoreML/MLMultiArray.h
_F32, _F64, _I32 = 65568, 65600, 131104


def _runtime():
    try:
        from rubicon.objc import ObjCClass, ns_from_py, py_from_ns
    except ImportError as exc:
        raise NeuralError("Install the pure-Python bridge: pip install rubicon-objc") from exc

    # Load the framework explicitly before asking the ObjC runtime for classes.
    # If iOS disallows dynamic loading in this app, fail cleanly.
    try:
        ctypes.CDLL("/System/Library/Frameworks/CoreML.framework/CoreML")
        return ObjCClass, ns_from_py, py_from_ns
    except (OSError, ValueError) as exc:
        raise NeuralError("Core ML framework is unavailable in this Python host") from exc


def available():
    """Whether the Core ML Objective-C classes are accessible (not ANE usage)."""
    try:
        cls, _, _ = _runtime()
        cls("MLModel")
        return True
    except (NeuralError, RuntimeError, LookupError, OSError):
        return False


def _shape_and_flat(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [], [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError("multiarray input must be a number or nested numeric lists")
    if not value:
        return [0], []
    s0, f0 = _shape_and_flat(value[0])
    flat = list(f0)
    for item in value[1:]:
        shape, part = _shape_and_flat(item)
        if shape != s0:
            raise ValueError("ragged multiarray input")
        flat.extend(part)
    return [len(value)] + s0, flat


def _to_feature(value, description, ObjCClass, ns):
    t = int(description.type)
    fv = ObjCClass("MLFeatureValue")
    if t == _DOUBLE:
        return fv.featureValueWithDouble_(float(value))
    if t == _INT:
        return fv.featureValueWithInt64_(int(value))
    if t == _STRING:
        return fv.featureValueWithString_(ns(str(value)))
    if t != _ARRAY:
        raise NeuralError("Only scalar, string and multiarray inputs are supported")
    _, flat = _shape_and_flat(value)
    constraint = description.multiArrayConstraint
    shape = [int(n) for n in constraint.shape]
    count = 1
    for dim in shape:
        count *= dim
    if len(flat) != count:
        raise ValueError(f"expected {count} elements for multiarray shape {shape}, got {len(flat)}")
    dtype = int(constraint.dataType)
    if dtype not in (_F32, _F64, _I32):
        raise NeuralError(f"multiarray dtype {dtype} is not supported yet")
    # Let Core ML allocate the right strides; index through its Objective-C API.
    arr = ObjCClass("MLMultiArray").alloc().initWithShape_dataType_error_(
        ns(shape), dtype, None
    )
    if arr is None:
        raise NeuralError("could not allocate MLMultiArray")
    for index, item in enumerate(flat):
        arr.setObject_atIndexedSubscript_(ns(int(item) if dtype == _I32 else float(item)), index)
    return fv.featureValueWithMultiArray_(arr)


def _from_feature(feature, py):
    t = int(feature.type)
    if t == _DOUBLE:
        return float(feature.doubleValue)
    if t == _INT:
        return int(feature.int64Value)
    if t == _STRING:
        return py(feature.stringValue)
    if t == _DICT:
        return py(feature.dictionaryValue)
    if t == _ARRAY:
        arr = feature.multiArrayValue
        shape = [int(n) for n in arr.shape]
        is_int = int(arr.dataType) == _I32
        values = [
            int(py(arr.objectAtIndexedSubscript_(i))) if is_int
            else float(py(arr.objectAtIndexedSubscript_(i)))
            for i in range(int(arr.count))
        ]
        def reshape(dims, offset):
            if not dims:
                return values[offset], offset + 1
            result = []
            for _ in range(dims[0]):
                item, offset = reshape(dims[1:], offset)
                result.append(item)
            return result, offset
        return reshape(shape, 0)[0]
    raise NeuralError(f"unsupported output feature type {t}")


class Model:
    def __init__(self, native, path, units, objc, ns, py):
        self._model, self.path, self.compute_units = native, str(path), units
        self._objc, self._ns, self._py = objc, ns, py

    def _descriptions(self, key):
        features = getattr(self._model.modelDescription, key)
        result = {}
        for name in features:
            desc = features.objectForKey_(name)
            info = {"type": int(desc.type), "optional": bool(desc.isOptional)}
            if int(desc.type) == _ARRAY:
                info["shape"] = [int(n) for n in desc.multiArrayConstraint.shape]
                info["dtype"] = int(desc.multiArrayConstraint.dataType)
            result[self._py(name)] = info
        return result

    @property
    def inputs(self):
        return self._descriptions("inputDescriptionsByName")

    @property
    def outputs(self):
        return self._descriptions("outputDescriptionsByName")

    def predict(self, inputs=None, **kwargs):
        values = dict(inputs or {})
        values.update(kwargs)
        desc = self._model.modelDescription.inputDescriptionsByName
        known = set(self.inputs)
        if set(values) - known:
            raise NeuralError(f"unknown inputs: {sorted(set(values) - known)}")
        missing = [name for name, info in self.inputs.items()
                   if name not in values and not info["optional"]]
        if missing:
            raise NeuralError(f"missing required inputs: {missing}")
        native = {}
        for name, value in values.items():
            native[name] = _to_feature(
                value, desc.objectForKey_(self._ns(name)), self._objc, self._ns
            )
        # ns_from_py cannot recursively convert a dict containing ObjCInstance
        # MLFeatureValue objects. Build a native NSDictionary explicitly.
        native_dict = self._objc("NSMutableDictionary").dictionary()
        for name, feature in native.items():
            native_dict.setObject_forKey_(feature, self._ns(name))
        provider = self._objc("MLDictionaryFeatureProvider").alloc().initWithDictionary_error_(
            native_dict, None
        )
        if provider is None:
            raise NeuralError("failed to create input provider")
        result = self._model.predictionFromFeatures_error_(provider, None)
        if result is None:
            raise NeuralError("Core ML prediction failed (no NSError details available)")
        # Iterate native NSSet via NSArray indexing; Rubicon may leave NSSet
        # wrapped as ObjCInstance even after py_from_ns conversion.
        names = result.featureNames.allObjects()
        output = {}
        for index in range(int(names.count)):
            name = names.objectAtIndex_(index)
            output[str(self._py(name))] = _from_feature(
                result.featureValueForName_(name), self._py
            )
        return output

    __call__ = predict


def load(path, compute_units="all"):
    """Load a compiled .mlmodelc. Other formats are compiled with Core ML first."""
    if compute_units not in _COMPUTE:
        raise ValueError(f"compute_units must be one of {tuple(_COMPUTE)}")
    path = Path(os.fspath(path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    objc, ns, py = _runtime()
    MLModel = objc("MLModel")
    url = objc("NSURL").fileURLWithPath_(ns(str(path)))
    if path.suffix.lower() != ".mlmodelc":
        url = MLModel.compileModelAtURL_error_(url, None)
        if url is None:
            raise NeuralError(f"could not compile Core ML model: {path}")
    config = objc("MLModelConfiguration").new()
    config.computeUnits = _COMPUTE[compute_units]
    model = MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    if model is None:
        raise NeuralError(f"could not load Core ML model: {path}")
    return Model(model, path, compute_units, objc, ns, py)


def neural_engine_available():
    """Not reliably detectable on every iOS release; None means unknown.

    Using compute_units='all' or 'cpu_ne' requests eligible hardware but does
    not prove that the Neural Engine executed an inference.
    """
    return None


__all__ = ["load", "Model", "available", "neural_engine_available", "NeuralError"]

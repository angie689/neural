"""Generate a small two-layer ReLU Core ML neural network (binary XOR).

No coremltools required. Core ML protobuf wire encoding follows Apple's public
Model.proto, FeatureTypes.proto and NeuralNetwork.proto.
"""
from pathlib import Path
import struct


def vi(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def field(number, value):
    return vi(number * 8) + vi(value)


def blob(number, data):
    return vi(number * 8 + 2) + vi(len(data)) + data


def label(number, value):
    return blob(number, value.encode("utf-8"))


def floats(number, values):
    return blob(number, struct.pack("<" + "f" * len(values), *values))


def feature(name, size):
    array_type = blob(5, blob(1, vi(size)) + field(2, 65568))
    return label(1, name) + blob(3, array_type)


def layer(name, src, dest, kind, properties):
    return label(1, name) + label(2, src) + label(3, dest) + blob(kind, properties)


def dense(name, src, dest, inputs, outputs, weights, biases):
    params = (
        field(1, inputs) + field(2, outputs) + field(10, 1)
        + blob(20, floats(1, weights)) + blob(21, floats(1, biases))
    )
    return layer(name, src, dest, 140, params)


layers = [
    dense("dense1", "input", "hidden", 2, 2,
          [1.0, -1.0, -1.0, 1.0], [0.0, 0.0]),
    layer("relu", "hidden", "activated", 130, blob(10, b"")),
    dense("dense2", "activated", "output", 2, 1, [1.0, 1.0], [0.0]),
]
network = b"".join(blob(1, item) for item in layers) + field(5, 1)
description = blob(1, feature("input", 2)) + blob(10, feature("output", 1))
model = field(1, 4) + blob(2, description) + blob(500, network)

destination = Path(__file__).parent / "XORNeuralNetwork.mlmodel"
destination.write_bytes(model)
print(f"Created {destination} ({len(model)} bytes)")

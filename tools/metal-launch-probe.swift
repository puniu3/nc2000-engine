import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal device unavailable")
}
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void rng_probe(device const ulong* input [[buffer(0)]], device ulong* output [[buffer(1)]],
                      constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    ulong seed = input[id];
    for (uint draw = 0; draw < 8; draw++) seed = seed * 0x5D588B656C078965ul + 0x269EC3ul;
    output[id] = seed;
}
"""
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "rng_probe")!)
let batches = [1, 32, 256, 4096, 65536]
var rows: [[String: Any]] = []
func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
@inline(never) func cpu(_ input: [UInt64], _ output: inout [UInt64]) {
    for i in input.indices {
        var seed = input[i]
        for _ in 0..<8 { seed = seed &* 0x5D588B656C078965 &+ 0x269EC3 }
        output[i] = seed
    }
}
for n in batches {
    let input = device.makeBuffer(length: n * 8, options: .storageModeShared)!
    let output = device.makeBuffer(length: n * 8, options: .storageModeShared)!
    let inPtr = input.contents().bindMemory(to: UInt64.self, capacity: n)
    let outPtr = output.contents().bindMemory(to: UInt64.self, capacity: n)
    var cpuInput = [UInt64](repeating: 0, count: n)
    var expected = cpuInput
    var gpuTimes: [Double] = []
    var cpuTimes: [Double] = []
    var kernelTimes: [Double] = []
    var checksum: UInt64 = 0
    for repeatIndex in 0..<110 {
        for i in 0..<n { cpuInput[i] = UInt64(i) &+ UInt64(repeatIndex) &* 137 }
        let ct = now()
        cpu(cpuInput, &expected)
        let cpuNs = Double(now() - ct)
        let gt = now()
        for i in 0..<n { inPtr[i] = cpuInput[i] }
        let cb = queue.makeCommandBuffer()!
        let encoder = cb.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var count = UInt32(n)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(n, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        precondition(cb.status == .completed, "GPU command failed")
        for i in 0..<n { checksum = checksum &+ outPtr[i] }
        let gpuNs = Double(now() - gt)
        for i in 0..<n { precondition(outPtr[i] == expected[i], "PRNG mismatch") }
        if repeatIndex >= 10 {
            cpuTimes.append(cpuNs)
            gpuTimes.append(gpuNs)
            kernelTimes.append((cb.gpuEndTime - cb.gpuStartTime) * 1e9)
        }
    }
    cpuTimes.sort(); gpuTimes.sort(); kernelTimes.sort()
    rows.append(["batch": n, "cpu_median_ns": cpuTimes[50], "gpu_e2e_median_ns": gpuTimes[50],
                 "gpu_e2e_p95_ns": gpuTimes[95], "gpu_e2e_min_ns": gpuTimes[0], "gpu_command_median_ns": kernelTimes[50],
                 "exact": true, "checksum": String(checksum)])
}
let result: [String: Any] = ["schema": "nc2000-metal-launch-v1", "device": device.name,
                            "workload": "8 exact Gen5 LCG draws per lane; launch lower-bound probe, not a battle engine",
                            "rows": rows]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)

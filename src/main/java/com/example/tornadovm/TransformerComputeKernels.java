package com.example.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class TransformerComputeKernels {

    /**
     * Default constructor for the TransformerComputeKernels class.
     */
    public TransformerComputeKernels() {
    }

    public static void emptyTaskToForceCopyIn(FloatArray buffer) {
        float dummy = buffer.get(0);
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, dummy);
        }
    }

    /**
     * Performs RMS (Root Mean Square) normalization using parallel reduction.
     * This is a two-phase reduction: first within work groups, then across work groups.
     *
     * Phase 1: Each work group computes a partial sum of squares
     * Phase 2: First thread combines all partial sums and computes normalization factor
     *
     * @param context Kernel execution context
     * @param output Array to store partial sums and final normalization factor
     * @param x Input array to normalize
     * @param size Number of elements to process
     * @param ermsNorm Epsilon value for numerical stability (epsilon * epsilon)
     * @param localMemSize Size of local memory allocation (work group size)
     */
    public static void reductionOneBlockWithLayer(KernelContext context, FloatArray output, FloatArray x, int size, float ermsNorm, int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(groupId + 1, localX[0]);
        }

        // Only the first thread in the first workgroup computes the final normalization factor
//        if (gid == 0) {
//            // Combine partial sums from all workgroups
//            float ss = 0.0f;
//            for (int i = 1; i <= (size / localMemSize); i++) {  // Assuming 8 workgroups
//                ss += output.get(i);
//            }
//
//            ss /= size;
//            ss += ermsNorm;
//            ss = 1.0f / TornadoMath.sqrt(ss);
//            output.set(0, ss);  // Store the final scale factor
//        }
    }

    /**
     * Applies the computed normalization factor to scale weights.
     * This is the second phase of RMS normalization.
     *
     * @param context Kernel execution context
     * @param output Array for normalized output
     * @param weights Weight values to normalize
     * @param temp Temporary array containing a normalization factor at index 0
     */
    public static void reductionOneBlock2WithLogits(KernelContext context, FloatArray output, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        output.set(gid, weights.get(gid) * (ss * output.get(gid)));
    }

    /**
     * Optimized matrix-vector multiplication for Q8_0 quantized weights.
     * Q8_0 format: 8-bit quantization with FP16 scale per 32-element block.
     *
     * Block format:
     * - Bytes 0-1: Float16 scale value (big-endian)
     * - Bytes 2-33: 32 quantized values (int8)
     *
     * Optimization features:
     * - Loop unrolling (16x factor)
     * - Vectorization (4 elements per iteration)
     * - Scale caching to avoid redundant decompression
     * - Fused multiply-add for accumulation
     *
     * @param context Kernel execution context
     * @param thisx Quantized matrix in row-major order
     * @param that Vector to multiply
     * @param out Output vector (result)
     * @param dim1 Vector dimension
     */
    public static void matmulTornadoQ8Optimized(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Block size used in quantization
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values
        final int UNROLL_FACTOR = 16; // Increased unroll factor for better performance
        final int VECTOR_SIZE = 4; // Process 4 elements at once with vectorization

        int idx = context.globalIdx;
        float result = 0f;
        int thisOffset = idx * dim1;

        // Cache last block index and scale to avoid redundant decoding
        int lastBlockIndex = -1;
        float cachedScale = 0f;

        // Early calculation of block boundaries to reduce in-loop calculations
        int numFullUnrolls = dim1 / UNROLL_FACTOR;
        int remainingStart = numFullUnrolls * UNROLL_FACTOR;

        // Pre-calculate first block index to potentially save work in the loop
        int firstIndex = thisOffset;
        int firstBlockIndex = firstIndex / BLOCK_SIZE;
        int firstBlockOffset = firstBlockIndex * BYTES_PER_BLOCK;

        // Initial scale calculation outside the loop
        int scaleByte1 = thisx.get(firstBlockOffset) & 0xFF;
        int scaleByte2 = thisx.get(firstBlockOffset + 1) & 0xFF;
        short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
        cachedScale = decodeFloat16Fast(scaleFloat16);
        lastBlockIndex = firstBlockIndex;

        // Main loop with increased unrolling
        for (int j = 0; j < numFullUnrolls; j++) {
            int baseIdx = j * UNROLL_FACTOR;

            // Process elements in groups of UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k += VECTOR_SIZE) {
                // Process VECTOR_SIZE elements in each iteration
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    int index = thisOffset + baseIdx + k + v;
                    int blockIndex = index / BLOCK_SIZE;

                    // Only decode scale if we're in a new block
                    if (blockIndex != lastBlockIndex) {
                        int blockOffset = blockIndex * BYTES_PER_BLOCK;
                        int newScaleByte1 = thisx.get(blockOffset) & 0xFF;
                        int newScaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                        short newScaleFloat16 = (short) ((newScaleByte2 << 8) | newScaleByte1);
                        cachedScale = decodeFloat16Fast(newScaleFloat16);
                        lastBlockIndex = blockIndex;
                    }

                    int withinBlockIndex = index % BLOCK_SIZE;
                    int blockOffset = blockIndex * BYTES_PER_BLOCK;

                    // Read quantized value
                    byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

                    // Dequantize and accumulate
                    result = fma(quantized * cachedScale, that.get(baseIdx + k + v), result);
                }
            }
        }

        // Handle remaining elements
        for (int j = remainingStart; j < dim1; j++) {
            int index = thisOffset + j;
            int blockIndex = index / BLOCK_SIZE;

            // Only decode scale if we're in a new block
            if (blockIndex != lastBlockIndex) {
                int blockOffset = blockIndex * BYTES_PER_BLOCK;
                int scaleByte11 = thisx.get(blockOffset) & 0xFF;
                int scaleByte22 = thisx.get(blockOffset + 1) & 0xFF;
                short scaleFloat166 = (short) ((scaleByte22 << 8) | scaleByte11);
                cachedScale = decodeFloat16Fast(scaleFloat166);
                lastBlockIndex = blockIndex;
            }

            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Read quantized value
            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

            // Dequantize and accumulate
            result = fma(quantized * cachedScale, that.get(j), result);
        }

        out.set(idx, result);
    }

    /**
     * Optimized matrix-vector multiplication for Q4_0 quantized weights.
     * Q4_0 format: 4-bit quantization with FP16 scale per 32-element block.
     *
     * Block format:
     * - Bytes 0-1: Float16 scale value (big-endian)
     * - Bytes 2-17: 16 bytes storing 32 packed 4-bit values
     *
     * Each byte stores two 4-bit values:
     * - Lower nibble: elements 0, 2, 4, ...
     * - Upper nibble: elements 1, 3, 5, ...
     *
     * Q4_0 uses signed 4-bit values with -8 offset (range: -8 to 7)
     *
     * @param context Kernel execution context
     * @param thisx Quantized matrix in row-major order
     * @param that Vector to multiply
     * @param out Output vector (result)
     * @param dim1 Vector dimension
     */
    public static void matmulTornadoQ4Optimized(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Block size for Q4_0
        final int BYTES_PER_BLOCK = 2 + BLOCK_SIZE / 2; // 2 bytes for scale + 16 bytes for packed values
        final int UNROLL_FACTOR = 16; // Unroll factor for better performance
        final int VECTOR_SIZE = 4; // Process 4 elements at once with vectorization

        int idx = context.globalIdx;
        float result = 0f;
        int thisOffset = idx * dim1;

        // Cache last block index and scale to avoid redundant decoding
        int lastBlockIndex = -1;
        float cachedScale = 0f;

        // Early calculation of block boundaries to reduce in-loop calculations
        int numFullUnrolls = dim1 / UNROLL_FACTOR;
        int remainingStart = numFullUnrolls * UNROLL_FACTOR;

        // Pre-calculate first block index to potentially save work in the loop
        int firstIndex = thisOffset;
        int firstBlockIndex = firstIndex / BLOCK_SIZE;
        int firstBlockOffset = firstBlockIndex * BYTES_PER_BLOCK;

        // Initial scale calculation outside the loop
        int scaleByte1 = thisx.get(firstBlockOffset) & 0xFF;
        int scaleByte2 = thisx.get(firstBlockOffset + 1) & 0xFF;
        short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
        cachedScale = decodeFloat16Fast(scaleFloat16);
        lastBlockIndex = firstBlockIndex;

        // Main loop with increased unrolling
        for (int j = 0; j < numFullUnrolls; j++) {
            int baseIdx = j * UNROLL_FACTOR;

            // Process elements in groups of UNROLL_FACTOR
            for (int k = 0; k < UNROLL_FACTOR; k += VECTOR_SIZE) {
                // Process VECTOR_SIZE elements in each iteration
                for (int v = 0; v < VECTOR_SIZE; v++) {
                    int index = thisOffset + baseIdx + k + v;
                    int blockIndex = index / BLOCK_SIZE;

                    // Only decode scale if we're in a new block
                    if (blockIndex != lastBlockIndex) {
                        int blockOffset = blockIndex * BYTES_PER_BLOCK;
                        int newScaleByte1 = thisx.get(blockOffset) & 0xFF;
                        int newScaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                        short newScaleFloat16 = (short) ((newScaleByte2 << 8) | newScaleByte1);
                        cachedScale = decodeFloat16Fast(newScaleFloat16);
                        lastBlockIndex = blockIndex;
                    }

                    int withinBlockIndex = index % BLOCK_SIZE;
                    int blockOffset = blockIndex * BYTES_PER_BLOCK;

                    // Extract Q4 value from packed byte
                    byte quant;
                    if (withinBlockIndex < BLOCK_SIZE / 2) {
                        // Lower nibble
                        quant = (byte) (thisx.get(blockOffset + 2 + withinBlockIndex) & 0x0F);
                    } else {
                        // Upper nibble
                        quant = (byte) ((thisx.get(blockOffset + 2 + withinBlockIndex - BLOCK_SIZE / 2) >>> 4) & 0x0F);
                    }

                    // Apply Q4 offset and scale
                    quant -= 8;  // Q4 uses -8 offset

                    // Dequantize and accumulate
                    result = fma(quant * cachedScale, that.get(baseIdx + k + v), result);
                }
            }
        }

        // Handle remaining elements
        for (int j = remainingStart; j < dim1; j++) {
            int index = thisOffset + j;
            int blockIndex = index / BLOCK_SIZE;

            // Only decode scale if we're in a new block
            if (blockIndex != lastBlockIndex) {
                int blockOffset = blockIndex * BYTES_PER_BLOCK;
                int scaleByte11 = thisx.get(blockOffset) & 0xFF;
                int scaleByte22 = thisx.get(blockOffset + 1) & 0xFF;
                short scaleFloat166 = (short) ((scaleByte22 << 8) | scaleByte11);
                cachedScale = decodeFloat16Fast(scaleFloat166);
                lastBlockIndex = blockIndex;
            }

            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Extract Q4 value from packed byte
            byte quant;
            if (withinBlockIndex < BLOCK_SIZE / 2) {
                // Lower nibble
                quant = (byte) (thisx.get(blockOffset + 2 + withinBlockIndex) & 0x0F);
            } else {
                // Upper nibble
                quant = (byte) ((thisx.get(blockOffset + 2 + withinBlockIndex - BLOCK_SIZE / 2) >>> 4) & 0x0F);
            }

            // Apply Q4 offset and scale
            quant -= 8;  // Q4 uses -8 offset

            // Dequantize and accumulate
            result = fma(quant * cachedScale, that.get(j), result);
        }

        out.set(idx, result);
    }

    /**
     * Fast decoder for IEEE 754 half-precision (Float16) floating point format.
     * Converts 16-bit encoded values to 32-bit float.
     * Float16 format:
     * - Bit 15: Sign bit
     * - Bits 14-10: Exponent (5 bits)
     * - Bits 9-0: Mantissa/Fraction (10 bits)
     * Special cases:
     * - Exponent all 1s: Infinity (mantissa 0) or NaN (mantissa non-zero)
     * - Exponent all 0s: Zero (mantissa 0) or denormalized (mantissa non-zero)
     *
     * @param value 16-bit encoded Float16 value
     * @return Decoded 32-bit float value
     */
    private static float decodeFloat16Fast(short value) {
        // Split the components
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        // Handle special cases with direct returns for common values
        if (exp == 0x1F) {
            return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }

        if (exp == 0) {
            if (frac == 0) {
                return sign == 0 ? 0.0f : -0.0f;
            }
            // Optimize denormalized numbers with precomputed constant
            float result = frac * 5.9604645E-8f; // Precomputed 2^-24
            return sign == 0 ? result : -result;
        }

        // Normal case - optimize with fewer operations
        float result = 1.0f + (frac / 1024.0f);

        // Use bitshift instead of pow for integer powers of 2
        if (exp < 15) {
            int shift = 15 - exp;
            result /= (1 << shift);
        } else {
            int shift = exp - 15;
            result *= (1 << shift);
        }

        return sign == 0 ? result : -result;
    }


    /**
     * Fused multiply-add operation: a * b + c
     * This is typically implemented as a single instruction on modern hardware,
     * providing better performance and numeric precision than separate operations.
     *
     * @param a First factor
     * @param b Second factor
     * @param c Value to add
     * @return Result of a * b + c
     */
    private static float fma(float a, float b, float c) {
        return a * b + c;
    }

}

package com.example.tornadovm;

import com.example.core.model.GGMLType;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.tensors.Float16;

import java.util.stream.IntStream;

public class TornadoVMCompute {
    public static final boolean TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
    public static final long WORKGROUP = Long.parseLong(System.getProperty("llama.workgroup", "256"));

    public TornadoVMCompute() {
    }

    public static void matmulTornadoQ4(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = GGMLType.Q4_0.getBlockSize(); // Q4 block size
        final int BYTES_PER_BLOCK = GGMLType.Q4_0.getTypeSize(); // The block size in bytes for Q4

        int idx = context.globalIdx;

        float result = 0f;
        int thisOffset = idx * dim1;

        for (int j = 0; j < dim1; j++) {
            int index = thisOffset + j;

            // Calculate block position and within-block index
            int blockIndex = index / BLOCK_SIZE;
            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Decode quantized value and scale
            float dequantizedValue = decodeQ4(thisx, blockOffset, withinBlockIndex);

            // Multiply the dequantized value by the corresponding element in 'that'
            result += dequantizedValue * that.get(j);
        }

        // Store the result in the output array
        out.set(idx, result);
    }

    public static void matmulTornadoQ8(KernelContext context, ByteArray thisx, FloatArray that, FloatArray out, int dim1) {
        final int BLOCK_SIZE = 32; // Assuming this is the block size used in quantization
        final int BYTES_PER_BLOCK = Float16.BYTES + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values

        int idx = context.globalIdx;

        float result = 0f;
        int thisOffset = idx * dim1;

        for (int j = 0; j < dim1; j++) {
            int index = thisOffset + j;
            // Calculate block position
            int blockIndex = index / BLOCK_SIZE;
            int withinBlockIndex = index % BLOCK_SIZE;
            int blockOffset = blockIndex * BYTES_PER_BLOCK;

            // Read scale (float16) for this block
            int scaleByte1 = thisx.get(blockOffset) & 0xFF;
            int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
            short scaleFloat16 = (short)((scaleByte2 << 8) | scaleByte1);
            float scale = decodeFloat16(scaleFloat16);

            // Read quantized value
            byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);

            // Dequantize and multiply
            result += (quantized * scale) * that.get(j);
        }

        out.set(idx, result);

    }

    private static float decodeFloat16(short value) {
        int sign = (value & 0x8000) >>> 15;
        int exp = (value & 0x7C00) >>> 10;
        int frac = value & 0x03FF;

        // Handle special cases
        if (exp == 0x1F) return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        if (exp == 0) {
            if (frac == 0) return sign == 0 ? 0.0f : -0.0f;
            float result = frac * pow2(-24);
            return sign == 0 ? result : -result;
        }

        float result = 1.0f + (frac / 1024.0f);
        result *= pow2(exp - 15);
        return sign == 0 ? result : -result;
    }

    /**
     * Compute 2^n efficiently
     */
    private static float pow2(int n) {
        if (n >= 0) {
            if (n < 31) {
                return (float)(1 << n);
            }
            return Float.POSITIVE_INFINITY;
        }
        if (n > -150) {
            return 1.0f / (1 << -n);
        }
        return 0.0f;
    }

    private static float decodeQ4(ByteArray thisx, int blockOffset, int withinBlockIndex) {
        // Read the scale (Float16 format)
        int scaleByte1 = thisx.get(blockOffset) & 0xFF;
        int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
        short scaleFloat16 = (short) ((scaleByte2 << 8) | scaleByte1);
        float scale = decodeFloat16(scaleFloat16);

        // Determine the byte in which the quantized value is stored
        byte quant;
        if (withinBlockIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            // The lower 4 bits of the byte
            quant = (byte) (thisx.get(blockOffset + Float16.BYTES + withinBlockIndex) & 0x0F);
        } else {
            // The higher 4 bits of the byte
            quant = (byte) ((thisx.get(blockOffset + Float16.BYTES + withinBlockIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }

        // Dequantize by shifting the value to the range [-8, 7]
        quant -= 8;

        // Return the dequantized value
        return quant * scale;
    }


    public static void matmul(ByteArray thisx, FloatArray that, FloatArray out, int dim0, int dim1) {
        final int BLOCK_SIZE = 32; // Assuming this is the block size used in quantization
        final int BYTES_PER_BLOCK = Float16.BYTES + BLOCK_SIZE; // 2 bytes for scale + block_size bytes for values

        IntStream.range(0, dim0).parallel().forEach(i -> {
            float result = 0f;
            int thisOffset = i * dim1;

            for (int j = 0; j < dim1; j++) {
                int index = thisOffset + j;
                // Calculate block position
                int blockIndex = index / BLOCK_SIZE;
                int withinBlockIndex = index % BLOCK_SIZE;
                int blockOffset = blockIndex * BYTES_PER_BLOCK;

                // Read scale (float16) for this block
                int scaleByte1 = thisx.get(blockOffset) & 0xFF;
                int scaleByte2 = thisx.get(blockOffset + 1) & 0xFF;
                short scaleFloat16 = (short)((scaleByte2 << 8) | scaleByte1);
                float scale = decodeFloat16(scaleFloat16);

                // Read quantized value
                byte quantized = thisx.get(blockOffset + 2 + withinBlockIndex);
                //                byte quantized = thisx.get(blockOffset + Float16.BYTES + withinBlockIndex);

                // Dequantize and multiply
                result += (quantized * scale) * that.get(j);
            }

            out.set(i, result);
        });
    }

    private static float getFloatFromByteArray(int index, ByteArray data) {
        // Direct read of two bytes at the index position - no multiplication by BYTES_PER_FLOAT16
        int byte1 = data.get(index) & 0xFF;
        int byte2 = data.get(index + 1) & 0xFF;
        short float16Value = (short)((byte2 << 8) | byte1);

        return decodeFloat16(float16Value);
    }

    public static void matmudl(ByteArray thisx, FloatArray that, FloatArray out, int dim0, int dim1) {
        IntStream.range(0, dim0).parallel().forEach(i -> {
            float result = 0f;
            int thisOffset = i * dim1;
            for (int j = 0; j < dim1; j++) {
                //                result += thisx.get(thisOffset + j) * that.get(j);
                result += getFloatFromByteArray(thisOffset + j, thisx) * that.get(j);
            }
            out.set(i, result);
        });
    }


    public static void reduceSquareSums(KernelContext context, FloatArray x, FloatArray reduce) {
        int globalIdx = context.globalIdx;
        int localIdx = context.localIdx;
        int localGroupSize = context.localGroupSizeX;
        int groupID = context.groupIdx; // Expose Group ID

        float[] localA = context.allocateFloatLocalArray((int) 256);
        localA[localIdx] = x.get(globalIdx) * x.get(globalIdx);
        for (int stride = (localGroupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (localIdx < stride) {
                localA[localIdx] += localA[localIdx + stride];
            }
        }
        if (localIdx == 0) {
            reduce.set(groupID, localA[0]);
        }
    }

    public static void finalSum(KernelContext context, FloatArray reduce, int size, float eps) {
        int globalIdx = context.globalIdx;

        float sum = 0.0f;
        if (globalIdx == 0) {
            for (int i = 0; i < size; i++) {
                sum += reduce.get(i);
            }
        }

        float ss = sum / (float) size;
        ss += eps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        reduce.set(0,ss);
    }

    public static void normalizeAndScale(
            KernelContext context, FloatArray x,
            FloatArray weight, FloatArray scalingFactorBuffer,
            int size) {

        int globalIdx = context.globalIdx;

        if (globalIdx < size) {
            float scaledValue = weight.get(globalIdx) * (scalingFactorBuffer.get(0) * x.get(globalIdx));
            x.set(globalIdx, scaledValue);
        }
    }

    public static void normalizeAndScale2(
            KernelContext context, FloatArray out,
            FloatArray x,
            FloatArray weight, FloatArray scalingFactorBuffer,
            int size) {

        int globalIdx = context.globalIdx;

        if (globalIdx < size) {
            float scaledValue = weight.get(globalIdx) * (scalingFactorBuffer.get(0) * x.get(globalIdx));
            out.set(globalIdx, scaledValue);
        }
    }

    public static void addInPlace(FloatArray input, FloatArray output) {
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            // Perform element-wise addition
            float result = output.get(i) + input.get(i);
            output.set(i, result);
        }
    }


    public static void multiplyInPlace(FloatArray input, FloatArray output) {
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            // Perform element-wise multiplication
            float result = output.get(i) * input.get(i);
            output.set(i, result);
        }
    }


    public static void mapInPlace(FloatArray input) {
        for (@Parallel int i = 0; i < input.getSize(); i++) {
            // Apply the transformation: value -> value / (1.0 + exp(-value))
            float value = input.get(i);
            float result = value / (1.0f + TornadoMath.exp(-value));
            input.set(i, result);
        }
    }

    public static void matrixVectorSimple(FloatArray x, FloatArray xout, FloatArray w, int n, int d) {
        for (@Parallel int i = 0; i <xout.getSize(); i++) {
            float val = 0f;
            for (int j = 0; j < x.getSize(); j++) {
                val += w.get(i * n + j) * x.get(j);
            }
            xout.set(i, val);
        }
    }


}

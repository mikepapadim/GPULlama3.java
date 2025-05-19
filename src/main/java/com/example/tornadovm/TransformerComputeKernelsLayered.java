package com.example.tornadovm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

public class TransformerComputeKernelsLayered {

    /**
     * Default constructor for the TransformerComputeKernelsLayered class.
     */
    public TransformerComputeKernelsLayered() {
    }

    /**
     * Performs RMS (Root Mean Square) normalization using parallel reduction.
     * This is the first phase of RMS normalization that computes the variance
     * and scaling factor across all work groups.
     *
     * Algorithm:
     * 1. Each thread computes square of its input element
     * 2. Work group performs parallel reduction of squares
     * 3. Partial sums stored per work group
     * 4. First thread combines all partial sums and computes normalization factor
     *
     * @param context Kernel execution context
     * @param output Array to store partial sums and final normalization factor
     * @param x Input array to normalize
     * @param size Number of elements to process
     * @param ermsNorm Epsilon value squared for numerical stability
     * @param localMemSize Size of local memory allocation (must match work group size)
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
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i <= (size / localMemSize); i++) {  // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);  // Store the final scale factor
        }
    }

    /**
     * Applies the computed normalization factor to input and weight elements.
     * This is the second phase of RMS normalization.
     *
     * Formula: output[i] = weight[i] * (normalizationFactor * x[i])
     *
     * @param context Kernel execution context
     * @param output Array for normalized output
     * @param x Input values to normalize
     * @param weights Weight values for each element
     * @param temp Temporary array containing normalization factor at index 0
     */
    public static void reductionOneBlock2WithLayer(KernelContext context, FloatArray output, FloatArray x, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;

        float ss = temp.get(0);
        output.set(gid, weights.get(gid) * (ss * x.get(gid)));
    }

    /**
     * Copies keys and values into the key-value cache for attention computation.
     * Enables efficient access to past key-value pairs during autoregressive generation.
     *
     * Cache layout: [layer][position][dimension]
     * - Each layer has its own key and value cache
     * - Each position in sequence has a key and value vector
     *
     * @param destKeyCache Destination array for key cache
     * @param srcKey Source keys to copy
     * @param destValueCache Destination array for value cache
     * @param srcValue Source values to copy
     * @param positioNlayer Array containing current position
     * @param kvDim Dimension of key/value vectors
     * @param layer Current transformer layer index
     * @param contextLength Maximum sequence length
     */
    public static void copyToCache(FloatArray destKeyCache, FloatArray srcKey, FloatArray destValueCache, FloatArray srcValue, IntArray positioNlayer, int kvDim, int layer, int contextLength) {

        int position = positioNlayer.get(0);
        int loff = layer * contextLength * kvDim;
        int destOffset = loff + position * kvDim;

        for (@Parallel int i = 0; i < srcValue.getSize(); i++) {
            destKeyCache.set(destOffset + i, srcKey.get(i));
            destValueCache.set(destOffset + i, srcValue.get(i));
        }
    }

    /**
     * Applies Rotary Position Encoding (RoPE) to query and key vectors.
     * RoPE rotates pairs of dimensions based on their position in the sequence,
     * enabling the model to learn relative positional information.
     *
     * For each pair of dimensions (2*i, 2*i+1):
     * - Compute rotation angle based on position and frequency
     * - Apply 2D rotation to the pair
     *
     * @param context Kernel execution context
     * @param positionHolder Array containing current position
     * @param sq Query vectors to rotate
     * @param sk Key vectors to rotate
     * @param kv_dim Dimension of key/value vectors
     * @param head_size Dimension of each attention head
     */
    public static void ropeRotation(KernelContext context, IntArray positionHolder, FloatArray sq, FloatArray sk, int kv_dim, int head_size) {
        int i = context.globalIdx * 2;

        int head_dim = i % head_size;
        // 50000.0f vs 10000.0f
        float freq = 1.0f / TornadoMath.pow(50000.0f, head_dim / (float) head_size);
        float val = positionHolder.get(0) * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only

        // Rotate query vector
        float v0q = sq.get(i);
        float v1q = sq.get(i + 1);
        sq.set(i, v0q * fcr - v1q * fci);
        sq.set(i + 1, v0q * fci + v1q * fcr);

        // Rotate key vector if needed
        if (rotn > 1 && i < sk.getSize()) {
            float v0k = sk.get(i);
            float v1k = sk.get(i + 1);
            sk.set(i, v0k * fcr - v1k * fci);
            sk.set(i + 1, v0k * fci + v1k * fcr);
        }

    }

    /**
     * Orchestrates parallel multi-head attention computation across all heads.
     * Each head processes attention independently in parallel.
     *
     * Attention computation:
     * 1. Compute attention scores (Q·K)
     * 2. Apply softmax for attention weights
     * 3. Compute weighted sum of values (attention·V)
     *
     * @param q Query vectors for all heads
     * @param key_cache Cached key vectors
     * @param value_cache Cached value vectors
     * @param xb Output buffer for attention results
     * @param nHeads Number of attention heads
     * @param headSize Dimension of each head
     * @param kvDim Total key/value dimension
     * @param kvMul Key/value head multiplier for grouped-query attention
     * @param seqLen Current sequence length
     * @param positionHolder Array containing position and layer info
     * @param wrapAtt Buffer for attention weights
     * @param layer Current transformer layer
     * @param contextLength Maximum context length
     */
    public static void processHeadsParallel(FloatArray q, FloatArray key_cache, FloatArray value_cache, FloatArray xb, int nHeads, int headSize, int kvDim, int kvMul, int seqLen,
            IntArray positionHolder, FloatArray wrapAtt, int layer, int contextLength) {

        int pos = positionHolder.get(0);
        int loff = layer * contextLength * kvDim;

        // Parallelize computation across attention heads
        for (@Parallel int h = 0; h < nHeads; h++) {
            // Process each head in parallel
            processHeadTornado(q, key_cache, value_cache, xb, h, headSize, kvDim, kvMul, loff, pos, wrapAtt);
        }
    }

    /**
     * Computes attention for a single head.
     * Implements scaled dot-product attention with softmax normalization.
     *
     * Steps:
     * 1. Compute attention scores: Q·K / sqrt(head_size)
     * 2. Apply softmax (with max subtraction for numerical stability)
     * 3. Compute weighted sum of values
     *
     * @param allQ All query vectors
     * @param key_cache Cached keys
     * @param value_cache Cached values
     * @param allXb Output buffer
     * @param h Head index to process
     * @param headSize Dimension per head
     * @param kvDim Key/value dimension
     * @param kvMul Key multiplier for grouped attention
     * @param loff Layer offset in cache
     * @param pos Current position
     * @param wrapAtt Attention weights buffer
     */
    private static void processHeadTornado(FloatArray allQ, FloatArray key_cache, FloatArray value_cache, FloatArray allXb, int h, int headSize, int kvDim, int kvMul, long loff, int pos,
            FloatArray wrapAtt) {

        // Base index for this head's attention weights
        int headOffset = h * (pos + 1);

        // STEP 1: Calculate attention scores for all timesteps
        for (int t = 0; t <= pos; t++) {
            int kvHeadIdx = h / kvMul;
            int keyOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);

            float score = 0.0f;
            for (int i = 0; i < headSize; i++) {
                score += allQ.get(h * headSize + i) * key_cache.get(keyOffset + i);
            }
            score = score / TornadoMath.sqrt(headSize);

            // Store in attention buffer
            wrapAtt.set(headOffset + t, score);
        }

        // STEP 2: Find max score for softmax stability
        float maxScore = wrapAtt.get(headOffset);
        for (int t = 1; t <= pos; t++) {
            float val = wrapAtt.get(headOffset + t);
            if (val > maxScore) {
                maxScore = val;
            }
        }

        // STEP 3: Compute exponentials and sum
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            float expScore = TornadoMath.exp(wrapAtt.get(idx) - maxScore);
            wrapAtt.set(idx, expScore);
            sum += expScore;
        }

        // STEP 4: Normalize
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos + 1));
        for (int t = 0; t <= pos; t++) {
            int idx = headOffset + t;
            wrapAtt.set(idx, wrapAtt.get(idx) * normFactor);
        }

        // STEP 5: Compute weighted sum of values for each dimension
        for (int i = 0; i < headSize; i++) {
            float weightedSum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                int kvHeadIdx = h / kvMul;
                int valueOffset = (int) (loff + t * kvDim + kvHeadIdx * headSize);
                weightedSum += wrapAtt.get(headOffset + t) * value_cache.get(valueOffset + i);
            }
            allXb.set(h * headSize + i, weightedSum);
        }
    }

    public static void processHeadsFlashAttention(
            KernelContext context,
            FloatArray q,
            FloatArray key_cache,
            FloatArray value_cache,
            FloatArray xb,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int layer,
            int contextLength) {

        // Thread and workgroup information
        int tid = context.localIdx;
        int h = context.groupIdx;  // Each workgroup processes one head
        int localSize = context.localGroupSizeX;

        // Early exit if this workgroup is beyond our head count
        // This relies on the kernel being launched with nHeads workgroups.
        if (h >= nHeads) return;

        int pos = positionHolder.get(0);
        int loff = layer * contextLength * kvDim;
        int kvHeadIdx = h / kvMul;
        int BLOCK_SIZE_C = 4;

        // Allocate shared memory for tiled computation
        float[] q_shared = context.allocateFloatLocalArray(headSize);
        float[] k_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] v_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] s_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C);
        float[] shared_tile_max_holder = context.allocateFloatLocalArray(1); // FIX: For broadcasting tile max

        // Thread-local accumulators for online softmax
        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;

        // Thread-local output accumulation
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        // Load query vector into shared memory
        for (int i = tid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }

        context.localBarrier();

        // Process sequence in tiles
        for (int tileC = 0; tileC <= pos; tileC += BLOCK_SIZE_C) {
            int tileEnd = Math.min(tileC + BLOCK_SIZE_C - 1, pos);

            // Load key and value vectors for this tile
            // Each thread loads a portion of the K and V vectors for the tile
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int k_v_idx_in_tile = tIdxInSeq - tileC; // 0, 1, 2, or 3 for this tile
                int tileMemOffset = k_v_idx_in_tile * headSize;
                for (int d = 0; d < headSize; d++) {
                    int kvCacheAbsolutePos = tIdxInSeq;
                    int kvOffset = loff + kvCacheAbsolutePos * kvDim + kvHeadIdx * headSize + d;
                    k_tile[tileMemOffset + d] = key_cache.get(kvOffset);
                    v_tile[tileMemOffset + d] = value_cache.get(kvOffset);
                }
            }

            context.localBarrier();

            // Compute attention scores for this tile
            // Each thread computes one score for the tile
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int score_idx_in_tile = tIdxInSeq - tileC; // 0, 1, 2, or 3 for this tile

                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += q_shared[d] * k_tile[score_idx_in_tile * headSize + d];
                }
                score /= TornadoMath.sqrt(headSize);
                s_tile[score_idx_in_tile] = score;
            }

            context.localBarrier();

            // Find max score in this tile (all threads compute it redundantly over the small s_tile)
            float tileLocalMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i <= tileEnd - tileC; i++) { // Iterate over valid scores in s_tile
                if (s_tile[i] > tileLocalMax) {
                    tileLocalMax = s_tile[i];
                }
            }

            // Broadcast max to all threads via shared memory
            if (tid == 0) {
                shared_tile_max_holder[0] = tileLocalMax; // FIX: Use dedicated holder
            }
            context.localBarrier();
            float currentTileMax = shared_tile_max_holder[0]; // FIX: Read from dedicated holder

            // Determine if we need to rescale previous results
            float newMax = Math.max(maxScore, currentTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= scale;
                }
            }
            maxScore = newMax;

            // Process each key-value pair using original scores from s_tile
            // All threads iterate over all scores in the current tile
            for (int t_idx_in_s_tile = 0; t_idx_in_s_tile <= tileEnd - tileC; t_idx_in_s_tile++) {
                // s_tile[t_idx_in_s_tile] now correctly refers to the original score
                float expScore = TornadoMath.exp(s_tile[t_idx_in_s_tile] - maxScore);
                sumExp += expScore;

                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * v_tile[t_idx_in_s_tile * headSize + d];
                }
            }
            context.localBarrier(); // Ensure all threads finish with s_tile, k_tile, v_tile before next tile load
        }

        // Normalize and write final results
        float normFactor = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f; // Avoid division by zero, return 0 if sumExp is 0
        for (int d = tid; d < headSize; d += localSize) {
            xb.set(h * headSize + d, output[d] * normFactor);
        }
    }

    /**
     * Performs optimized matrix-vector multiplication where each work group
     * processes one row of the matrix.
     *
     * Algorithm:
     * 1. Each work group handles one output dimension
     * 2. Threads in work group compute partial dot products
     * 3. Parallel reduction yields final row result
     *
     * @param context Kernel execution context
     * @param x Input vector
     * @param hb Output vector
     * @param w Weight matrix (row-major)
     * @param n Input dimension
     * @param d Output dimension
     * @param localWorkGroupSize Number of threads per work group
     */
    public static void matrixVectorGeneric(KernelContext context, FloatArray x, FloatArray hb, FloatArray w, int n, int d, int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }
        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n, d);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            hb.set(rowId, sum);
        }
    }

    /**
     * Matrix-vector multiplication with residual connection.
     * Combines regular matrix multiplication with addition of existing values.
     *
     * Formula: hb[i] = hb[i] + w[i]·x
     *
     * @param context Kernel execution context
     * @param x Input vector
     * @param hb Input/output vector (contains residual, receives result)
     * @param w Weight matrix
     * @param n Input dimension
     * @param d Output dimension
     * @param localWorkGroupSize Work group size
     */
    public static void matrixVectorGenericWithResidual(KernelContext context, FloatArray x, FloatArray hb, FloatArray w, int n, int d, int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        int localSize = localWorkGroupSize;

        // Early exit if this workgroup is beyond our output dimension
        if (rowId >= d) {
            return;
        }

        float sum = matrixVectorRowMajorOptimized(context, localSize, x, w, n, d);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float result = hb.get(rowId) + sum;
            hb.set(rowId, result);
        }
    }

    /**
     * Fused feed-forward network with SiLU activation and GLU gating.
     * Implements the SwiGLU variant used in LLaMA-style models.
     *
     * Formula: FFN(x) = SiLU(x·W1) ⊙ (x·W3)
     * where ⊙ denotes element-wise multiplication
     *
     * @param context Kernel execution context
     * @param x Input vector
     * @param hb Output buffer
     * @param w1 First feed-forward weight matrix
     * @param w3 Third feed-forward weight matrix (gate)
     * @param n Input dimension
     * @param d Hidden dimension
     * @param localWorkGroupSize Work group size
     */
    public static void fusedFeedForwardWithSiLUAndGLUActivation(KernelContext context, FloatArray x, FloatArray hb, FloatArray w1, FloatArray w3, int n, int d, int localWorkGroupSize) {
        // One row per workgroup (not per thread)
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum1 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w1, n, d);
        float sum3 = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w3, n, d);

        // Thread 0 in each workgroup writes the final result
        if (localId == 0) {
            float silu = siluActivation(sum1);  // Using the new SiLU method
            float result = silu * sum3;
            hb.set(rowId, result);
        }
    }

    /**
     * Gaussian Error Linear Unit (GELU) activation function.
     * Approximation formula: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     *
     * @param x Input value
     * @return Activated value
     */
    public static float geluActivation(float x) {
        float x3 = x * x * x;
        return 0.5f * x * (1.0f + TornadoMath.tanh((0.797885f * (x + 0.044715f * x3))));
    }

    /**
     * Sigmoid-weighted Linear Unit (SiLU) activation function.
     * Also known as Swish activation.
     *
     * Formula: SiLU(x) = x * σ(x) = x / (1 + e^(-x))
     *
     * @param x Input value
     * @return Activated value
     */
    public static float siluActivation(float x) {
        return x * (1.0f / (1.0f + TornadoMath.exp(-x)));
    }

    /**
     * Optimized row-major matrix-vector multiplication for a single row.
     * Uses parallel reduction within a work group to compute one dot product.
     *
     * Algorithm:
     * 1. Each thread computes partial dot product
     * 2. Partial results stored in local memory
     * 3. Tree-based reduction combines partial results
     * 4. Returns final dot product for the row
     *
     * @param context Kernel execution context
     * @param localSize Work group size
     * @param x Input vector
     * @param w Weight matrix row
     * @param n Input dimension
     * @param d Output dimension
     * @return Dot product result for this row
     */
    public static float matrixVectorRowMajorOptimized(KernelContext context, int localSize, FloatArray x, FloatArray w, int n, int d) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Allocate local memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localSize);

        int rowOffset = rowId * n;

        // Each thread calculates partial dot product
        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int matrixIdx = rowOffset + j;
            partialSum += w.get(matrixIdx) * x.get(j);
        }

        // Store partial sum in local memory
        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        return localSum[0];
    }
}

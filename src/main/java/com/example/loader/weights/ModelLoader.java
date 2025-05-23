package com.example.loader.weights;

import com.example.LlamaApp;
import com.example.aux.Timer;
import com.example.core.model.GGMLType;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.F16FloatTensor;
import com.example.core.model.tensor.FloatTensor;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.core.model.tensor.Q4_0FloatTensor;
import com.example.core.model.tensor.Q8_0FloatTensor;
import com.example.core.types.Pair;
import com.example.inference.engine.impl.Configuration;
import com.example.inference.engine.impl.Llama;
import com.example.inference.operation.RoPE;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class ModelLoader {
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";

    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    public static Llama loadModel(Path ggufPath, int contextLength, boolean loadWeights) throws IOException {
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        return loadModel(fileChannel, gguf, contextLength, loadWeights);
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) throws IOException {
        try (var ignored = Timer.log("Load LlaMa model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            Vocabulary vocabulary = Vocabulary.loadVocabulary(metadata);
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            Configuration config = new Configuration((int) metadata.get("llama.embedding_length"), (int) metadata.get("llama.feed_forward_length"), (int) metadata.get("llama.block_count"),
                    (int) metadata.get("llama.attention.head_count"),

                    metadata.containsKey("llama.attention.head_count_kv") ? (int) metadata.get("llama.attention.head_count_kv") : (int) metadata.get("llama.attention.head_count"),

                    vocabulary.size(), (int) metadata.get("llama.context_length"), (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)).withContextLength(contextLength);

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            return new Llama(config, tokenizer, weights);
        }
    }

    public static Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        RopeConfig ropeConfig = new RopeConfig(8.0f,         // scaleFactor
                1.0f,                    // loFreqFactor
                3.0f,                    // hiFreqFactor
                8192                     // oldContextLength
        );

        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength,      // Maximum sequence length the model can process
                config.headSize,           // Dimension of each attention head
                config.ropeTheta,          // Base frequency parameter (typically 10000.0)
                ropeScaling,               // Whether to apply frequency scaling (determined by model type)
                ropeConfig.scaleFactor,    // Scale factor for extending context length (NTK-aware scaling)
                ropeConfig.loFreqFactor,   // Low frequency scaling factor for better long-range dependencies
                ropeConfig.hiFreqFactor,   // High frequency scaling factor for preserving local precision
                ropeConfig.oldContextLength // Original context length the model was trained with
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault("output.weight", tokenEmbeddings);

        if (LlamaApp.USE_TORNADOVM) {
            System.out.println("Loading model weights in TornadoVM format (converting " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    private static Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Weights(
                // Load directly to TornadoVM format
                loadTensorAsFloatArray(tokenEmbeddings), loadArrayAsFloatArrayFromBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()), FloatArray.fromArray(ropeFreqs.second()), loadTensorAsHalfFloatArray(outputWeight), outputWeight.ggmlType());
    }

    /**
     * Creates weights in standard format only
     */
    private static Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Weights(loadQuantized(tokenEmbeddings), loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), toFloatBuffer(tensorEntries.get("output_norm.weight")),
                FloatBuffer.wrap(ropeFreqs.first()), FloatBuffer.wrap(ropeFreqs.second()), loadQuantized(outputWeight), outputWeight.ggmlType());
    }

    private static Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens = IntStream.range(0, specialTokensList.size()).boxed().collect(Collectors.toMap(i -> specialTokensList.get(i), i -> baseTokens + i));

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            //            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatArray[] loadArrayAsFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static HalfFloatArray[] loadArrayAsHalfFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        HalfFloatArray[] array = new HalfFloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsHalfFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatArray floatBufferToFloatArray(GGMLTensorEntry tensorEntry) {
        if (tensorEntry.ggmlType() == GGMLType.F32) {
            FloatBuffer buffer = tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            return FloatArray.fromFloatBuffer(buffer);
        } else {
            throw new UnsupportedOperationException("Conversion to FloatArray from " + tensorEntry.ggmlType());
        }
    }

    public static FloatArray[] loadArrayAsFloatArrayFromBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = floatBufferToFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static ByteArray createByteArrayFromTensor(GGMLTensorEntry entry) {
        FloatTensor tensor = loadQuantized(entry);
        return ByteArray.fromSegment(tensor.asMemorySegment());
    }

    public static FloatArray loadTensorAsFloatArray(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.F32) {
            // For F32, we can directly create FloatArray from memory
            FloatBuffer buffer = entry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            FloatArray array = new FloatArray(buffer.remaining());
            for (int i = 0; i < buffer.remaining(); i++) {
                array.set(i, buffer.get());
            }
            return array;
        } else {
            // For quantized formats, we need to load through FloatTensor
            FloatTensor tensor = loadQuantized(entry);
            FloatArray array = new FloatArray(tensor.size());
            for (int i = 0; i < tensor.size(); i++) {
                array.set(i, tensor.getFloat(i));
            }
            return array;
        }
    }

    public static HalfFloatArray loadTensorAsHalfFloatArray(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.F32) {
            System.out.println("Loading F32 tensor as HalfFloatArray");
            return null;
        } else {
            // For quantized formats, we need to load through FloatTensor
            FloatTensor tensor = loadQuantized(entry);
            HalfFloatArray array = new HalfFloatArray(tensor.size());
            for (int i = 0; i < tensor.size(); i++) {
                HalfFloat x = new HalfFloat(tensor.getFloat(i));
                array.set(i, x);
            }
            return array;
        }
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }

    // Helper class to encapsulate RoPE configuration parameters
    private static class RopeConfig {
        final float scaleFactor;
        final float loFreqFactor;
        final float hiFreqFactor;
        final int oldContextLength;

        RopeConfig(float scaleFactor, float loFreqFactor, float hiFreqFactor, int oldContextLength) {
            this.scaleFactor = scaleFactor;
            this.loFreqFactor = loFreqFactor;
            this.hiFreqFactor = hiFreqFactor;
            this.oldContextLength = oldContextLength;
        }
    }

}

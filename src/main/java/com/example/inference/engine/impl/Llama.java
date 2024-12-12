package com.example.inference.engine.impl;

import com.example.aux.Parallel;
import com.example.aux.Tuple2;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.Sampler;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMCompute;
import com.example.tornadovm.TornadoVMLayerPlanner;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {

    public State createNewState() {
        State state = new State(configuration());
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static FloatTensor forward(Llama model, State state, int token, int position,  ArrayList<Tuple2<TornadoExecutionPlan, GridScheduler>> tornadoVMListOfPlans) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);

            // qkv matmuls for this position

            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.get(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.get(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads, h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            if (!TornadoVMCompute.TORNADOVM) {
//            if (true) {
                ffnLayerJava(l, state, dim, config, weights);
            } else {
//                ffnLayerJava(l, state, dim, config, weights);
                ffnLayerTornadoVM(state, tornadoVMListOfPlans.get(l));
            }
        }


        if(TornadoVMCompute.TORNADOVM) {
//            state.wrapXFloat.getSegment().set(ValueLayout.ADDRESS,0, state.x.asMemorySegment());
            state.wrapXFloat.getSegment().copyFrom(state.x.asMemorySegment());
            tornadoVMListOfPlans.get(tornadoVMListOfPlans.size()-1).getFirst().withGridScheduler(tornadoVMListOfPlans.get(tornadoVMListOfPlans.size()-1).getSecond()).execute();
            state.logits.asMemorySegment().copyFrom(state.wrapLogits.getSegment());
            state.x.asMemorySegment().copyFrom(state.wrapXFloat.getSegment());
//            state.logits.asMemorySegment().set(ValueLayout.ADDRESS,0, state.wrapLogits.getSegment());
//            state.x.asMemorySegment().set(ValueLayout.ADDRESS,0, state.wrapXFloat.getSegment());

        } else {
            rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps);
            weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);
        }

        return state.logits;
    }

    static void ffnLayerJava(int l, State state, int dim, Configuration config, Weights weights) {
        // final matmul to get the output of the attention
        weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

        // residual connection back into x
        state.x.addInPlace(state.xb2);

        // ffn rmsnorm
        rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], dim, config.rmsNormEps);

        //            System.out.println("x " + weights.w1.toString() + " " + weights.w2.toString() + " " + weights.w3.toString());
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim, dim);
        weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim, dim);

        // SwiGLU non-linearity
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

        // elementwise multiply with w3(x)
        state.hb.multiplyInPlace(state.hb2);

        // final matmul to get the output of the ffn
        weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim);

        // residual connection
        state.x.addInPlace(state.xb);
    }

    static void ffnLayerTornadoVM(State state, Tuple2<TornadoExecutionPlan, GridScheduler> tornadoVMFFNLayer) {
        state.wrapXFloat.getSegment().copyFrom(state.x.asMemorySegment());
        state.wrapHb.getSegment().copyFrom(state.hb.asMemorySegment());
        state.wrapHb2.getSegment().copyFrom(state.hb2.asMemorySegment());
        state.wrapXb.getSegment().copyFrom(state.xb.asMemorySegment());
        state.wrapXb2.getSegment().copyFrom(state.xb2.asMemorySegment());

        tornadoVMFFNLayer.getFirst().withGridScheduler(tornadoVMFFNLayer.getSecond()).execute();

        state.xb2.asMemorySegment().copyFrom(state.wrapXb2.getSegment());
        state.xb.asMemorySegment().copyFrom(state.wrapXb.getSegment());
        state.hb2.asMemorySegment().copyFrom(state.wrapHb2.getSegment());
        state.hb.asMemorySegment().copyFrom(state.wrapHb.getSegment());
        state.x.asMemorySegment().copyFrom(state.wrapXFloat.getSegment());
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {

        TornadoVMLayerPlanner tornadoVMLayerPlanner = new TornadoVMLayerPlanner(state, model);
        ArrayList< Tuple2<TornadoExecutionPlan, GridScheduler>>  tornadoVMPlans = tornadoVMLayerPlanner.setupAndGetTornadoVMExecutionPlans();

        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;



        for (int position = startPosition; position < maxTokens; ++position) {
            forward(model, state, token, position,tornadoVMPlans);
            startGen = System.nanoTime();
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    // log prompt token (different color?)
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    // log inferred token
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        int totalTokens = promptIndex + generatedTokens.size();

        System.err.printf("\n%n%.2f tokens/s (%d) [PrEval %.2f tokens/s (%d), TokGen %.2f tokens/s (%d)]%n",
                totalTokens / (elapsedNanos / 1_000_000_000.0), totalTokens,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());
        return generatedTokens;
    }

}


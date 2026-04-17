/**
 * CPU MLP for decoding splat features into opacity, covariance, and color.
 *
 * Matches the GSmlpModel / GSHashModel architecture in smol-gs:
 *   - mlp_opacity : [f_dim] → 1  (last layer activation: tanh, then clamp >= 0)
 *   - mlp_cov     : [f_dim] → 7  (no output activation)
 *                    output[:3]  = scale modifiers  → sigmoid → multiply exp(_scaling)
 *                    output[3:7] = quaternion        → normalize
 *   - mlp_color   : [f_dim] → 3  (last layer activation: sigmoid, in [0,1])
 *   - octree      : [3*recursion] → octreeEncodingDim (optional, no output activation)
 *
 * Full input to opacity/cov/color MLPs:
 *   cat([ simple_feat (featureDim),
 *         octree_feat (octreeEncodingDim, optional),
 *         ob_view     (3, unit vector from splat toward camera),
 *         ob_dist     (1, distance from splat to camera) ])
 *
 * Octree encoding (replicates SpaceFillingCurves + octree_encoder in gs_render.py):
 *   unit_coor    = round((xyz - zero_bound) / unit)          [N, 3]  integer
 *   binary_coor  = int_to_bits(unit_coor)  reshaped [N, 3*recursion]  float {0,1}
 *   octree_feat  = octree_mlp(binary_coor)                   [N, octreeEncodingDim]
 *
 * Weight JSON format (weights.json):
 * {
 *   "opacity": [ { "weights": [[...]], "biases": [...], "activation": "relu" }, ...
 *                { "weights": [[...]], "biases": [...], "activation": "tanh"  } ],
 *   "cov":     [ ... last layer: "none" ],
 *   "color":   [ ... last layer: "sigmoid" ],
 *   "octree":  [ ... last layer: "none" ]   // omit if octree encoder not used
 * }
 */

export interface MLPLayer {
    /** weights[outDim][inDim] */
    weights: number[][];
    biases: number[];
    activation: "relu" | "sigmoid" | "tanh" | "none";
}

export interface MLPWeights {
    /** 1 output: raw opacity. Last layer activation should be "tanh", then clamp >= 0. */
    opacity: MLPLayer[];
    /**
     * 7 outputs, no output activation:
     *   [0:3] scale modifiers  (sigmoid in post-processing × exp(_scaling))
     *   [3:7] quaternion       (normalized in post-processing)
     */
    cov: MLPLayer[];
    /** 3 outputs, RGB. Last layer activation should be "sigmoid". */
    color: MLPLayer[];
    /**
     * Optional octree encoder MLP.
     * Input dim = 3 * sfCurveRecursion, output dim = octreeEncodingDim.
     * Architecture: Linear(3*rec, 64) → ReLU → Linear(64, octreeDim), no output activation.
     * Omit if the model was trained without an octree encoder.
     */
    octree?: MLPLayer[];
}

/** Space-filling curve parameters loaded from meta_info['sf_curve']. */
export interface SFCurveParams {
    /** lower bound of the scene bounding box [x, y, z] */
    lower: Float32Array;
    /** upper bound of the scene bounding box [x, y, z] */
    upper: Float32Array;
    /** number of bits per dimension (e.g. 10 → 2^10 = 1024 cells per axis) */
    recursion: number;
}

// ─── activation functions ─────────────────────────────────────────────────────

function relu(x: number): number {
    return x > 0 ? x : 0;
}

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function applyActivation(x: number, act: MLPLayer["activation"]): number {
    switch (act) {
        case "relu":    return relu(x);
        case "sigmoid": return sigmoid(x);
        case "tanh":    return Math.tanh(x);
        case "none":    return x;
    }
}

// ─── core forward pass ────────────────────────────────────────────────────────

/**
 * Run one forward pass through a stack of layers.
 */
export function mlpForward(input: Float32Array | number[], layers: MLPLayer[]): Float32Array {
    let x: Float32Array = input instanceof Float32Array ? input : new Float32Array(input);

    for (const layer of layers) {
        const outDim = layer.biases.length;
        const out = new Float32Array(outDim);
        for (let o = 0; o < outDim; o++) {
            let sum = layer.biases[o];
            const w = layer.weights[o];
            for (let i = 0; i < x.length; i++) {
                sum += w[i] * x[i];
            }
            out[o] = applyActivation(sum, layer.activation);
        }
        x = out;
    }

    return x;
}

// ─── octree encoding ──────────────────────────────────────────────────────────

/**
 * Compute derived sf_curve quantities from the stored lower/upper bounds.
 * Matches SpaceFillingCurves.__init__ (including the tolerance expansion).
 */
export function buildSFCurveCache(params: SFCurveParams): {
    zeroBound: Float32Array;
    unit: Float32Array;
} {
    const unit      = new Float32Array(3);
    const zeroBound = new Float32Array(3);
    const cells     = Math.pow(2, params.recursion);

    for (let d = 0; d < 3; d++) {
        unit[d]      = (params.upper[d] - params.lower[d]) / cells;
        zeroBound[d] = params.lower[d] + 0.5 * unit[d];
    }
    return { zeroBound, unit };
}

/**
 * Replicate SpaceFillingCurves.continous2discrete + int_to_bits for a single splat.
 *
 * Returns a Float32Array of length (3 * recursion) with values in {0, 1},
 * matching the Python:
 *   unit_coor   = round((xyz - zero_bound) / unit)          [3]
 *   binary_coor = int_to_bits(unit_coor).reshape(3*recursion)  {0,1}
 *
 * Bit layout: [x_bits(high→low), y_bits(high→low), z_bits(high→low)]
 */
function computeBinaryCoor(
    px: number, py: number, pz: number,
    zeroBound: Float32Array,
    unit: Float32Array,
    recursion: number,
    out: Float32Array,   // pre-allocated, length = 3 * recursion
): void {
    const xyz = [px, py, pz];
    for (let d = 0; d < 3; d++) {
        const intVal = Math.round((xyz[d] - zeroBound[d]) / unit[d]);
        for (let b = 0; b < recursion; b++) {
            // shifts[b] = recursion - 1 - b  (high bit first, matches Python)
            out[d * recursion + b] = (intVal >> (recursion - 1 - b)) & 1;
        }
    }
}

// ─── batch decode ─────────────────────────────────────────────────────────────

export interface DecodedAttributes {
    /** uint8 RGBA, length = 4 * vertexCount */
    colors: Uint8Array;
    /** float [qw, qx, qy, qz] normalized, length = 4 * vertexCount */
    rotations: Float32Array;
    /**
     * float world-space scales = exp(_scaling) × sigmoid(mlp_cov[:3]),
     * length = 3 * vertexCount.
     */
    scales: Float32Array;
}

/**
 * Decode all splat features using the three (or four, with octree) MLPs.
 *
 * Matches `generate_neural_gaussians` in gs_render.py:
 *   opacity = clamp(tanh(mlp_opacity(input)), 0, 1)
 *   color   = sigmoid(mlp_color(input))   [built-in last activation]
 *   rot     = normalize(mlp_cov(input)[3:7])
 *   scaling = exp(_scaling) × sigmoid(mlp_cov(input)[:3])
 *
 * @param features      Float32Array [N × featureDim]  (_simple_feat)
 * @param logScales     Float32Array [N × 3]           (_scaling, log-space)
 * @param featureDim    Features per vertex
 * @param weights       MLP weight definitions (opacity, cov, color; optional octree)
 * @param sfCurve       Space-filling curve params (required when weights.octree is set)
 * @param cameraPos     Camera world position [x,y,z].
 *                      Pass null to bake with a neutral view direction.
 * @param positions     Float32Array [N × 3] (xyz). Required when cameraPos is provided.
 */
export function decodeFeatures(
    features: Float32Array,
    logScales: Float32Array,
    featureDim: number,
    weights: MLPWeights,
    sfCurve: SFCurveParams | null = null,
    cameraPos: Float32Array | null = null,
    positions: Float32Array | null = null,
): DecodedAttributes {
    const vertexCount = features.length / featureDim;
    const hasOctree   = !!(weights.octree && weights.octree.length > 0 && sfCurve);

    // Pre-compute sf_curve quantities once
    let sfCache: { zeroBound: Float32Array; unit: Float32Array } | null = null;
    let octreeInputDim = 0;
    let octreeInput: Float32Array | null = null;
    if (hasOctree && sfCurve) {
        sfCache        = buildSFCurveCache(sfCurve);
        octreeInputDim = 3 * sfCurve.recursion;
        octreeInput    = new Float32Array(octreeInputDim);
    }

    // Infer octree output dim from the last layer's bias length
    const octreeOutDim = (hasOctree && weights.octree)
        ? weights.octree[weights.octree.length - 1].biases.length
        : 0;

    // Full MLP input: [simple_feat | octree_feat | ob_view(3) | ob_dist(1)]
    const inputDim = featureDim + octreeOutDim + 4;
    const input    = new Float32Array(inputDim);

    const colors    = new Uint8Array(4 * vertexCount);
    const rotations = new Float32Array(4 * vertexCount);
    const scales    = new Float32Array(3 * vertexCount);

    for (let i = 0; i < vertexCount; i++) {
        // ── simple_feat ────────────────────────────────────────────────────
        input.set(features.subarray(i * featureDim, (i + 1) * featureDim), 0);

        // ── octree_feat ────────────────────────────────────────────────────
        if (hasOctree && sfCurve && sfCache && octreeInput && weights.octree && positions) {
            const px = positions[3 * i + 0];
            const py = positions[3 * i + 1];
            const pz = positions[3 * i + 2];
            computeBinaryCoor(px, py, pz, sfCache.zeroBound, sfCache.unit, sfCurve.recursion, octreeInput);
            const octreeFeat = mlpForward(octreeInput, weights.octree);
            input.set(octreeFeat, featureDim);
        }

        // ── ob_view + ob_dist ──────────────────────────────────────────────
        const viewOffset = featureDim + octreeOutDim;
        if (cameraPos !== null && positions !== null) {
            const px  = positions[3 * i + 0];
            const py  = positions[3 * i + 1];
            const pz  = positions[3 * i + 2];
            const dvx = px - cameraPos[0];
            const dvy = py - cameraPos[1];
            const dvz = pz - cameraPos[2];
            const dist = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) + 1e-6;
            input[viewOffset + 0] = dvx / dist;
            input[viewOffset + 1] = dvy / dist;
            input[viewOffset + 2] = dvz / dist;
            input[viewOffset + 3] = dist;
        } else {
            // neutral bake
            input[viewOffset + 0] = 0;
            input[viewOffset + 1] = 0;
            input[viewOffset + 2] = 0;
            input[viewOffset + 3] = 1;
        }

        // ── opacity ─────────────────────────────────────í───────────────────
        // Last layer of mlp_opacity uses Tanh; clamp to [0, 1].
        const opOut = mlpForward(input, weights.opacity);
        colors[4 * i + 3] = Math.round(Math.max(0, opOut[0]) * 255);

        // ── color ──────────────────────────────────────────────────────────
        // Last layer of mlp_color uses Sigmoid → already in [0, 1].
        const colOut = mlpForward(input, weights.color);
        colors[4 * i + 0] = Math.round(Math.min(1, Math.max(0, colOut[0])) * 255);
        colors[4 * i + 1] = Math.round(Math.min(1, Math.max(0, colOut[1])) * 255);
        colors[4 * i + 2] = Math.round(Math.min(1, Math.max(0, colOut[2])) * 255);

        // ── covariance → rotation + scale ─────────────────────────────────
        // mlp_cov has no output activation.
        // scale_rot[:3] = scale modifiers → sigmoid × exp(_scaling)
        // scale_rot[3:7] = unnormalized quaternion → normalize
        const covOut = mlpForward(input, weights.cov);
        scales[3 * i + 0] = Math.exp(logScales[3 * i + 0]) * sigmoid(covOut[0]);
        scales[3 * i + 1] = Math.exp(logScales[3 * i + 1]) * sigmoid(covOut[1]);
        scales[3 * i + 2] = Math.exp(logScales[3 * i + 2]) * sigmoid(covOut[2]);

        const qw = covOut[3], qx = covOut[4], qy = covOut[5], qz = covOut[6];
        const qlen = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) || 1;
        rotations[4 * i + 0] = qw / qlen;
        rotations[4 * i + 1] = qx / qlen;
        rotations[4 * i + 2] = qy / qlen;
        rotations[4 * i + 3] = qz / qlen;
    }

    return { colors, rotations, scales };
}

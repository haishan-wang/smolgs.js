import { SplatData } from "./SplatData";
import { MLPWeights, SFCurveParams, decodeFeatures } from "../utils/MLP";

/**
 * Holds raw per-splat data for the GSmlpModel / GSHashModel compressed format.
 *
 * Per-splat storage (matching conduct_encoding output):
 *   - positions  (x, y, z)        — float32[N×3]
 *   - logScales  (_scaling)        — float32[N×3]  log-space
 *   - features   (_simple_feat)    — float32[N×featureDim]
 *
 * Scene-level metadata:
 *   - sfCurve    lower, upper, recursion  (for octree encoding)
 *
 * At render time, these plus a camera position feed the three MLPs
 * (mlp_opacity, mlp_cov, mlp_color) — and optionally the octree encoder —
 * to recover opacity, rotation, scale, and color.
 *
 * Usage:
 *   const mlpData = SplatMLPData.Deserialize(buffer);
 *
 *   // Static bake (neutral view, no octree):
 *   const splatData = mlpData.decode(weights);
 *
 *   // Static bake with octree encoder:
 *   const splatData = mlpData.decode(weights);  // sfCurve is embedded in mlpData
 *
 *   // View-dependent color + octree (call each frame):
 *   const camPos = new Float32Array([cx, cy, cz]);
 *   const splatData = mlpData.decode(weights, camPos);
 *
 * Binary file layout (.splat-mlp):
 *   Bytes  0– 3   magic        uint32   0x534D4C50 ("SMLP")
 *   Bytes  4– 7   vertexCount  uint32
 *   Bytes  8–11   featureDim   uint32
 *   Bytes 12–15   octreeDim    uint32   (octreeEncodingDim; 0 = no octree encoder)
 *   Bytes 16–19   recursion    uint32   (sf_curve recursion, e.g. 10)
 *   Bytes 20–31   reserved     uint32[3]
 *   Bytes 32–43   lower_bound  float32[3]
 *   Bytes 44–55   upper_bound  float32[3]
 *   Bytes 56–..   positions    float32[N×3]
 *   Bytes ..–..   logScales    float32[N×3]
 *   Bytes ..–end  features     float32[N×featureDim]
 */
class SplatMLPData {
    static readonly Magic      = 0x534d4c50;
    static readonly HeaderSize = 56; // bytes before float data

    private _vertexCount: number;
    private _featureDim: number;
    private _positions: Float32Array;
    private _logScales: Float32Array;
    private _features: Float32Array;
    private _sfCurve: SFCurveParams | null;

    constructor(
        vertexCount: number,
        featureDim: number,
        positions: Float32Array,
        logScales: Float32Array,
        features: Float32Array,
        sfCurve: SFCurveParams | null = null,
    ) {
        this._vertexCount = vertexCount;
        this._featureDim  = featureDim;
        this._positions   = positions;
        this._logScales   = logScales;
        this._features    = features;
        this._sfCurve     = sfCurve;
    }

    get vertexCount() { return this._vertexCount; }
    get featureDim()  { return this._featureDim; }
    get positions()   { return this._positions; }
    get logScales()   { return this._logScales; }
    get features()    { return this._features; }
    get sfCurve()     { return this._sfCurve; }

    /**
     * Run MLP inference and produce a standard SplatData ready for rendering.
     *
     * When `weights.octree` is present AND `this.sfCurve` is set, the octree
     * encoding is computed automatically from `this.positions`.
     *
     * @param weights    Trained MLP weights (opacity, cov, color; optional octree)
     * @param cameraPos  Optional camera world position [x, y, z].
     *                   Omit for a static bake with neutral view direction.
     *                   Pass the current camera position each frame for correct
     *                   view-dependent colors.
     */
    decode(weights: MLPWeights, cameraPos: Float32Array | null = null): SplatData {
        const sfCurve  = this._sfCurve;
        const useOctree = !!(weights.octree && weights.octree.length > 0 && sfCurve);

        const { colors, rotations, scales } = decodeFeatures(
            this._features,
            this._logScales,
            this._featureDim,
            weights,
            useOctree ? sfCurve : null,
            cameraPos,
            (cameraPos !== null || useOctree) ? this._positions : null,
        );

        return new SplatData(
            this._vertexCount,
            new Float32Array(this._positions),
            rotations,
            scales,
            colors,
        );
    }

    // ── serialization ─────────────────────────────────────────────────────────

    serialize(): ArrayBuffer {
        const N      = this._vertexCount;
        const D      = this._featureDim;
        const hBytes = SplatMLPData.HeaderSize;           // 56
        const buffer = new ArrayBuffer(hBytes + (3 + 3 + D) * N * 4);
        const dv     = new DataView(buffer);

        // Fixed-size header (32 bytes of uint32 fields)
        dv.setUint32(0,  SplatMLPData.Magic,        true);
        dv.setUint32(4,  N,                          true);
        dv.setUint32(8,  D,                          true);
        // octreeDim: infer from sfCurve presence (0 = absent; actual dim in weights.json)
        dv.setUint32(12, 0,                          true);
        dv.setUint32(16, this._sfCurve?.recursion ?? 0, true);
        dv.setUint32(20, 0, true); // reserved
        dv.setUint32(24, 0, true); // reserved
        dv.setUint32(28, 0, true); // reserved

        // lower_bound + upper_bound (24 bytes of float32)
        for (let d = 0; d < 3; d++) {
            dv.setFloat32(32 + d * 4, this._sfCurve?.lower[d] ?? 0, true);
            dv.setFloat32(44 + d * 4, this._sfCurve?.upper[d] ?? 0, true);
        }

        // Float32 payload
        const f32 = new Float32Array(buffer, hBytes);
        f32.set(this._positions, 0);
        f32.set(this._logScales, N * 3);
        f32.set(this._features,  N * 6);

        return buffer;
    }

    static Deserialize(buffer: ArrayBuffer): SplatMLPData {
        const dv          = new DataView(buffer);
        const magic       = dv.getUint32(0,  true);
        if (magic !== SplatMLPData.Magic) {
            throw new Error(`Invalid .splat-mlp file (bad magic: 0x${magic.toString(16)})`);
        }
        const vertexCount = dv.getUint32(4,  true);
        const featureDim  = dv.getUint32(8,  true);
        const recursion   = dv.getUint32(16, true);

        let sfCurve: SFCurveParams | null = null;
        if (recursion > 0) {
            const lower = new Float32Array([
                dv.getFloat32(32, true),
                dv.getFloat32(36, true),
                dv.getFloat32(40, true),
            ]);
            const upper = new Float32Array([
                dv.getFloat32(44, true),
                dv.getFloat32(48, true),
                dv.getFloat32(52, true),
            ]);
            sfCurve = { lower, upper, recursion };
        }

        const hBytes    = SplatMLPData.HeaderSize;
        const posStart  = hBytes;
        const scaleStart = posStart   + vertexCount * 3 * 4;
        const featStart  = scaleStart + vertexCount * 3 * 4;

        const positions = new Float32Array(new Float32Array(buffer, posStart,   vertexCount * 3));
        const logScales = new Float32Array(new Float32Array(buffer, scaleStart, vertexCount * 3));
        const features  = new Float32Array(new Float32Array(buffer, featStart,  vertexCount * featureDim));

        return new SplatMLPData(vertexCount, featureDim, positions, logScales, features, sfCurve);
    }
}

export { SplatMLPData };

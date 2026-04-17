import type { Scene } from "../core/Scene";
import { Splat } from "../splats/Splat";
import { SplatMLPData } from "../splats/SplatMLPData";
import { MLPWeights } from "../utils/MLP";
import { initiateFetchRequest, loadDataIntoBuffer } from "../utils/LoaderUtils";

/**
 * Loader for the compact MLP-decoded splat format (.splat-mlp).
 *
 * Usage:
 *   // 1. Load MLP weights (your trained model, exported as JSON)
 *   const weights: MLPWeights = await fetch("model_weights.json").then(r => r.json());
 *
 *   // 2. Load the splat-mlp file and decode in one call (static bake, neutral view direction)
 *   const splat = await MLPLoader.LoadAsync("scene.splat-mlp", scene, weights, onProgress);
 *
 * For view-dependent color (re-decode when camera moves):
 *   const mlpData = await MLPLoader.LoadRawAsync("scene.splat-mlp");
 *   const camPos  = new Float32Array([cx, cy, cz]);
 *   const splat   = new Splat(mlpData.decode(weights, camPos));
 *   scene.addObject(splat);
 *   // On camera move: splat._data = mlpData.decode(weights, newCamPos); splat._data.changed = true;
 *
 * If you already have an ArrayBuffer (e.g. from a File input), use:
 *   const splat = MLPLoader.LoadFromArrayBuffer(buffer, scene, weights);
 */
class MLPLoader {
    static async LoadAsync(
        url: string,
        scene: Scene,
        weights: MLPWeights,
        onProgress?: (progress: number) => void,
        useCache: boolean = false,
    ): Promise<Splat> {
        const res = await initiateFetchRequest(url, useCache);
        const buffer = await loadDataIntoBuffer(res, onProgress);
        return this.LoadFromArrayBuffer(buffer.buffer, scene, weights);
    }

    static async LoadFromFileAsync(
        file: File,
        scene: Scene,
        weights: MLPWeights,
        onProgress?: (progress: number) => void,
    ): Promise<Splat> {
        const reader = new FileReader();
        let splat = new Splat();
        reader.onload = (e) => {
            splat = this.LoadFromArrayBuffer(e.target!.result as ArrayBuffer, scene, weights);
        };
        reader.onprogress = (e) => {
            onProgress?.(e.loaded / e.total);
        };
        reader.readAsArrayBuffer(file);
        await new Promise<void>((resolve) => {
            reader.onloadend = () => resolve();
        });
        return splat;
    }

    static LoadFromArrayBuffer(arrayBuffer: ArrayBufferLike, scene: Scene, weights: MLPWeights): Splat {
        const mlpData  = SplatMLPData.Deserialize(arrayBuffer as ArrayBuffer);
        const splatData = mlpData.decode(weights);
        const splat    = new Splat(splatData);
        scene.addObject(splat);
        return splat;
    }

    /**
     * Load only the raw feature data without running the MLP.
     * Useful when you want to defer decoding or swap weights later.
     */
    static async LoadRawAsync(
        url: string,
        onProgress?: (progress: number) => void,
        useCache: boolean = false,
    ): Promise<SplatMLPData> {
        const res = await initiateFetchRequest(url, useCache);
        const buffer = await loadDataIntoBuffer(res, onProgress);
        return SplatMLPData.Deserialize(buffer.buffer as ArrayBuffer);
    }
}

export { MLPLoader };

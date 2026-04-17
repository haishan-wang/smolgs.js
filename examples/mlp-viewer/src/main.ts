import * as SPLAT from "gsplat";

const canvas   = document.getElementById("canvas")  as HTMLCanvasElement;
const statusEl = document.getElementById("status")  as HTMLDivElement;

const renderer = new SPLAT.WebGLRenderer(canvas);
const scene    = new SPLAT.Scene();
const camera   = new SPLAT.Camera();
const controls = new SPLAT.OrbitControls(camera, canvas);

async function main() {
    // Both files are served from web_data/ via the vite.config.js publicDir setting.
    const weights: SPLAT.MLPWeights = await fetch("weights.json").then(r => r.json());

    // Load raw features (no MLP bake yet) so we can re-decode each frame.
    const mlpData = await SPLAT.MLPLoader.LoadRawAsync(
        "scene.splat-mlp",
        (progress: number) => {
            statusEl.textContent = `Loading… ${Math.round(progress * 100)}%`;
        },
    );

    // Initial bake with neutral view so the splat exists in the scene.
    const splat = new SPLAT.Splat(mlpData.decode(weights));
    scene.addObject(splat);

    statusEl.style.display = "none";
    const p = camera.position;
    splat.data = mlpData.decode(weights, new Float32Array([p.x, p.y, p.z]));

    // const handleResize = () => renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    // handleResize();
    // window.addEventListener("resize", handleResize);

    let needsDecode = false;
    let lastCamX = NaN, lastCamY = NaN, lastCamZ = NaN;

    const frame = () => {
        controls.update();

        const p = camera.position;
        if (p.x !== lastCamX || p.y !== lastCamY || p.z !== lastCamZ) {
            lastCamX = p.x; lastCamY = p.y; lastCamZ = p.z;
            needsDecode = false;
            console.log(`camera moved`);
        }

        if (needsDecode) {
            needsDecode = false;
            const t0 = performance.now();
            splat.data = mlpData.decode(weights, new Float32Array([p.x, p.y, p.z]));
            console.log(`mlpData.decode took ${(performance.now() - t0).toFixed(2)}ms`);
        }

        renderer.render(scene, camera);
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
}

main().catch(err => {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
});

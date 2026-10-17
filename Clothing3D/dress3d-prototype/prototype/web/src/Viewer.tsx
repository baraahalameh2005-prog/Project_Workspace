import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

interface Props {
  glbUrl: string | null;
  levels: { name: string; y_cm: number }[];
}

type View = "front" | "side" | "back" | "top";

export default function Viewer({ glbUrl, levels }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const ctx = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    model: THREE.Object3D | null;
    guides: THREE.Group;
  } | null>(null);
  const [wire, setWire] = useState(false);
  const [showLevels, setShowLevels] = useState(true);
  const [status, setStatus] = useState("No garment yet");

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(getComputedStyle(el).getPropertyValue("--viewer-bg").trim() || "#ecebe8");
    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
    camera.position.set(0, 0.6, 3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, 2.5, 3);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.7);
    rim.position.set(-2, 1.5, -2.5);
    scene.add(rim);
    const grid = new THREE.GridHelper(2, 20, 0x999999, 0xcccccc); // 10 cm cells
    scene.add(grid);
    const guides = new THREE.Group();
    scene.add(guides);
    ctx.current = { renderer, scene, camera, controls, model: null, guides };

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  // load model
  useEffect(() => {
    const c = ctx.current;
    if (!c || !glbUrl) return;
    setStatus("Loading model…");
    new GLTFLoader().load(
      glbUrl,
      (gltf) => {
        if (c.model) c.scene.remove(c.model);
        const root = gltf.scene;
        root.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.isMesh) {
            const mats = Array.isArray(m.material) ? m.material : [m.material];
            mats.forEach((mat) => (mat.side = THREE.DoubleSide));
          }
        });
        // centre horizontally; hem stays on the grid (y = 0)
        const box = new THREE.Box3().setFromObject(root);
        const centre = box.getCenter(new THREE.Vector3());
        root.position.set(-centre.x, 0, -centre.z);
        c.scene.add(root);
        c.model = root;
        frame("front");
        setStatus("");
      },
      undefined,
      (e) => setStatus(`Failed to load model: ${String(e)}`),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [glbUrl]);

  useEffect(() => {
    const c = ctx.current;
    if (!c?.model) return;
    c.model.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) {
        const mats = Array.isArray(m.material) ? m.material : [m.material];
        mats.forEach((mat) => ((mat as THREE.MeshStandardMaterial).wireframe = wire));
      }
    });
  }, [wire]);

  // measurement level guides
  useEffect(() => {
    const c = ctx.current;
    if (!c) return;
    c.guides.clear();
    if (!showLevels) return;
    levels.forEach(({ y_cm }, i) => {
      const geo = new THREE.RingGeometry(0.29, 0.3, 64);
      const mat = new THREE.MeshBasicMaterial({
        color: [0xd9480f, 0x1971c2, 0x2f9e44, 0x7048e8][i % 4],
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
      });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = y_cm / 100;
      c.guides.add(ring);
    });
  }, [levels, showLevels]);

  function frame(view: View) {
    const c = ctx.current;
    if (!c?.model) return;
    const box = new THREE.Box3().setFromObject(c.model);
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    const dist = (Math.max(size.y, size.x) / 2 / Math.tan((c.camera.fov * Math.PI) / 360)) * 1.25;
    const dirs: Record<View, THREE.Vector3> = {
      front: new THREE.Vector3(0, 0.08, 1),
      side: new THREE.Vector3(1, 0.08, 0),
      back: new THREE.Vector3(0, 0.08, -1),
      top: new THREE.Vector3(0, 1, 0.0001),
    };
    c.camera.position.copy(centre).add(dirs[view].normalize().multiplyScalar(dist));
    c.controls.target.copy(centre);
    c.controls.update();
  }

  return (
    <div className="viewer-wrap">
      <div className="viewer" ref={host} aria-label="3D garment viewer" />
      {status && <div className="viewer-status">{status}</div>}
      <div className="viewer-bar">
        {(["front", "side", "back", "top"] as View[]).map((v) => (
          <button key={v} onClick={() => frame(v)} disabled={!glbUrl}>
            {v}
          </button>
        ))}
        <label>
          <input type="checkbox" checked={wire} onChange={(e) => setWire(e.target.checked)} /> wireframe
        </label>
        <label>
          <input type="checkbox" checked={showLevels} onChange={(e) => setShowLevels(e.target.checked)} /> bust/waist/hip
          levels
        </label>
      </div>
      <p className="hint">Drag to rotate · scroll to zoom · right-drag to pan · grid = 10 cm</p>
    </div>
  );
}

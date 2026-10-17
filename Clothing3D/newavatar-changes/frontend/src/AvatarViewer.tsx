import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export type ViewerStatus =
  | { state: "empty" }
  | { state: "loading"; url: string }
  | { state: "displayed"; url: string }
  | { state: "failed"; url: string; error: string };

export type ViewMode = "body" | "face";

interface Props {
  url: string | null;
  view: ViewMode;
  onStatus: (status: ViewerStatus) => void;
}

interface Stage {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  model: THREE.Object3D | null;
}

function disposeObject(root: THREE.Object3D) {
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry?.dispose();
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const material of materials) {
      if (!material) continue;
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) value.dispose();
      }
      material.dispose();
    }
  });
}

/** Point the camera at the whole body or at the head (top part of the bounding box). */
function frame(stage: Stage, view: ViewMode) {
  const { camera, controls, model } = stage;
  if (!model) return;

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  let target: THREE.Vector3;
  let height: number;
  if (view === "face") {
    // Head is roughly the top 13 % of an adult body.
    height = size.y * 0.16;
    target = new THREE.Vector3(center.x, box.max.y - size.y * 0.065, center.z);
  } else {
    height = size.y;
    target = center;
  }

  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const fitHeight = height / (2 * Math.tan(vFov / 2));
  const fitWidth = Math.max(size.x, size.z) / (2 * Math.tan(vFov / 2) * camera.aspect);
  const distance = Math.max(fitHeight, view === "face" ? 0 : fitWidth) * 1.25 + size.z / 2;

  camera.near = Math.max(distance / 100, 0.01);
  camera.far = distance * 100;
  camera.position.set(target.x, target.y, target.z + distance);
  camera.updateProjectionMatrix();
  controls.target.copy(target);
  controls.update();
}

export default function AvatarViewer({ url, view, onStatus }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Stage | null>(null);
  const viewRef = useRef(view);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;
  viewRef.current = view;

  // Scene setup and teardown.
  useEffect(() => {
    const container = containerRef.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe9ebee);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(1.5, 3, 2.5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-2, 1.5, -1.5);
    scene.add(fill);
    // Feet are at y = 0 in the exported GLB.
    const grid = new THREE.GridHelper(4, 20, 0x9aa0a8, 0xc9cdd3);
    scene.add(grid);

    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
    camera.position.set(0, 1, 5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.9, 0);

    const stage: Stage = { renderer, scene, camera, controls, model: null };
    stageRef.current = stage;

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = container;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    // Low on memory, the browser can drop the WebGL context and the canvas goes
    // blank. preventDefault lets it be restored; three.js then redraws the scene.
    const onContextLost = (event: Event) => {
      event.preventDefault();
      console.warn("WebGL context lost; waiting for the browser to restore it.");
    };
    renderer.domElement.addEventListener("webglcontextlost", onContextLost);

    return () => {
      renderer.domElement.removeEventListener("webglcontextlost", onContextLost);
      renderer.setAnimationLoop(null);
      observer.disconnect();
      controls.dispose();
      if (stage.model) {
        scene.remove(stage.model);
        disposeObject(stage.model);
        stage.model = null;
      }
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
      stageRef.current = null;
    };
  }, []);

  // Load the requested GLB. The previous model stays visible until the new one
  // has loaded; if loading fails, it is removed so no outdated avatar remains on screen.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;

    if (!url) {
      onStatusRef.current({ state: "empty" });
      return;
    }

    let cancelled = false;
    onStatusRef.current({ state: "loading", url });

    new GLTFLoader()
      .loadAsync(url)
      .then((gltf) => {
        if (cancelled || stageRef.current !== stage) {
          disposeObject(gltf.scene);
          return;
        }
        if (stage.model) {
          stage.scene.remove(stage.model);
          disposeObject(stage.model);
        }
        stage.model = gltf.scene;
        stage.scene.add(gltf.scene);
        frame(stage, viewRef.current);
        onStatusRef.current({ state: "displayed", url });
      })
      .catch((err: unknown) => {
        if (cancelled || stageRef.current !== stage) return;
        if (stage.model) {
          stage.scene.remove(stage.model);
          disposeObject(stage.model);
          stage.model = null;
        }
        onStatusRef.current({ state: "failed", url, error: err instanceof Error ? err.message : String(err) });
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  // Reframe when the view mode changes.
  useEffect(() => {
    if (stageRef.current) frame(stageRef.current, view);
  }, [view]);

  return <div className="viewer" ref={containerRef} />;
}
